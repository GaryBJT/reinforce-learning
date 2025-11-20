# MAPPO 版 QoAR（保持原 Q-learning 外部接口不变）
# - NaN/Inf 防护 + 坏权重自愈（Xavier重置+删除坏ckpt）
# - update_q_value / get_best_next_hop / update_lq 弹性少参兼容
# - set_mappo_params / set_qlearning_params 支持缺参(含缺 c)且静默
# - 默认静默；将环境变量 QOAR_VERBOSE=1 可开启日志

import os
import numbers
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

VERBOSE = os.environ.get("QOAR_VERBOSE", "0") == "1"
# VERBOSE = 1

__all__ = [
    "MAPPOQoAR",
    "qlearning",
    "update_q_value",
    "update_lq",
    "get_best_next_hop",
    "set_qlearning_params",
    "set_mappo_params",
    "po_params",
    "set_po_params",
]

# 尽量避免和 ns-3 线程/OMP 冲突
try:
    torch.set_num_threads(1)
except Exception:
    pass


# ====== 小型 MLP ======
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(128, 128), out_dim=None):
        super().__init__()
        dims = [in_dim] + list(hidden)
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.out = nn.Linear(dims[-1], out_dim) if out_dim is not None else None

    def forward(self, x):
        for l in self.layers:
            x = F.relu(l(x))
        return self.out(x) if self.out is not None else x


# ====== MAPPO：Actor / Critic ======
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(128, 128)):
        super().__init__()
        self.net = MLP(obs_dim, hidden, act_dim)

    def forward(self, obs):
        logits = self.net(obs)  # [B, act_dim]
        return logits


class Critic(nn.Module):
    def __init__(self, gobs_dim, hidden=(128, 128)):
        super().__init__()
        self.v = MLP(gobs_dim, hidden, 1)

    def forward(self, gobs):
        return self.v(gobs).squeeze(-1)  # [B]


# ====== On-policy 缓冲（GAE）======
class OnPolicyBuf:
    def __init__(self, capacity, gamma=0.95, lam=0.95, device="cpu"):
        self.capacity = int(capacity)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.device = device
        self.clear()

    def clear(self):
        self.obs, self.gobs, self.act = [], [], []
        self.old_logp, self.rew, self.val, self.done = [], [], [], []

    def add(self, obs, gobs, act, logp, rew, val, done):
        self.obs.append(obs)
        self.gobs.append(gobs)
        self.act.append(act)
        self.old_logp.append(logp)
        self.rew.append(rew)
        self.val.append(val)
        self.done.append(done)

    def __len__(self):
        return len(self.rew)

    def gae(self, last_v=0.0):
        r = torch.tensor(self.rew, dtype=torch.float32, device=self.device)
        v = torch.tensor(self.val, dtype=torch.float32, device=self.device)
        d = torch.tensor(self.done, dtype=torch.float32, device=self.device)

        adv = torch.zeros_like(r)
        v_ext = torch.cat([v, torch.tensor([last_v], device=self.device)])
        gae = 0.0
        for t in reversed(range(len(r))):
            delta = r[t] + self.gamma * v_ext[t + 1] * (1 - d[t]) - v[t]
            gae = delta + self.gamma * self.lam * (1 - d[t]) * gae
            adv[t] = gae
        ret = adv + v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = torch.tensor(np.array(self.obs), dtype=torch.float32, device=self.device)
        gobs = torch.tensor(np.array(self.gobs), dtype=torch.float32, device=self.device)
        act = torch.tensor(np.array(self.act), dtype=torch.long, device=self.device)
        oldp = torch.tensor(np.array(self.old_logp), dtype=torch.float32, device=self.device)
        return obs, gobs, act, oldp, adv, ret


# ====== 主体：MAPPO QoAR ======
class MAPPOQoAR:
    """
    - 共享 Actor，集中式 Critic
    - 复用链路质量 a,b,c；LQ = a*sf + b*ef + c*bf
    - 对外接口兼容，但内部训练为 PPO（MAPPO）
    """

    def __init__(self, alpha=0.8, gamma=0.9, a=0.5, b=0.0, c=0.5, buffer_size=100, batch_size=50, replay_interval=10):
        self.set_parameters(alpha, gamma, a, b, c)

        use_cuda = os.environ.get("QOAR_USE_CUDA") == "1" and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        # 观测：onehot(current,H) + onehot(dest,H) + LQ统计(3)
        self.H = 64
        self.obs_dim = self.H * 2 + 3
        self.gobs_dim = self.obs_dim

        # 动作空间
        self.act_dim = 64
        self.action_map = {}            # next_hop -> idx
        self.inverse_action_map = {}    # idx -> next_hop

        # (current,dest) 可选动作集合
        self.state_action_set = defaultdict(set)

        # 节点索引与最近观测缓存
        self.node_map = {}
        self.last_obs_by_node = defaultdict(lambda: np.zeros(self.H * 2 + 3, dtype=np.float32))

        # 链路质量表
        self.link_quality = defaultdict(lambda: defaultdict(float))  # current -> next_hop -> lq

        # 网络
        self.actor = Actor(self.obs_dim, self.act_dim).to(self.device)
        self.critic = Critic(self.gobs_dim).to(self.device)
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=self.pi_lr)
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=self.vf_lr)

        # PPO / GAE
        self.clip_eps = 0.2
        self.lam = 0.95
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.epochs = 4
        self.mini_batch = 256

        self.train_batch = max(1024, int(buffer_size) * 20)
        self.buf = OnPolicyBuf(self.train_batch, gamma=self.gamma, lam=self.lam, device=self.device)
        self.update_counter = 0
        self.replay_interval = int(replay_interval)

        os.makedirs("models", exist_ok=True)
        self._load()

        # —— 权重体检与自愈：发现 NaN/Inf 就重置并删坏 ckpt ——
        if not self._model_finite(self.actor) or not self._model_finite(self.critic):
            if VERBOSE:
                print("[MAPPO] 检测到NaN/Inf权重，已重置并删除坏ckpt")
            self._reinit_actor_critic()
            try:
                os.remove(self._ckpt())
            except Exception:
                pass

    # ===== 参数设置 =====
    def set_parameters(self, alpha, gamma, a, b, c):
        if abs(a + b + c - 1.0) > 1e-6:
            raise ValueError("链路质量权重之和必须为1")
        self.a, self.b, self.c = float(a), float(b), float(c)
        self.gamma = float(gamma)
        self.pi_lr = float(alpha)
        self.vf_lr = float(max(1e-4, alpha))

    # ===== NaN/Inf 防护 & 自愈 =====
    def _sanitize(self, t: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)

    def _reinit_actor_critic(self):
        def init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.actor.apply(init)
        self.critic.apply(init)

    def _model_finite(self, model: nn.Module) -> bool:
        for p in model.parameters():
            if not torch.isfinite(p).all():
                return False
        return True

    # ===== 编码/聚合 =====
    def _idx(self, name):
        if name not in self.node_map:
            self.node_map[name] = len(self.node_map)
        return self.node_map[name] % self.H

    def _obs(self, current, dest):
        cur = np.zeros(self.H, dtype=np.float32)
        cur[self._idx(current)] = 1.0
        dst = np.zeros(self.H, dtype=np.float32)
        dst[self._idx(dest)] = 1.0

        lqs = list(self.link_quality[current].values())
        if len(lqs) == 0:
            stats = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            best = float(np.max(lqs))
            mean = float(np.mean(lqs))
            minv = float(np.min(lqs))
            stats = np.array([best, mean, minv], dtype=np.float32)
        obs = np.concatenate([cur, dst, stats], axis=0)
        self.last_obs_by_node[current] = obs
        return obs

    def _gobs(self):
        if not self.last_obs_by_node:
            return np.zeros(self.gobs_dim, dtype=np.float32)
        agg = np.sum(np.stack(list(self.last_obs_by_node.values()), axis=0), axis=0)
        return np.clip(agg, 0.0, 1.0).astype(np.float32)

    def _dist(self, obs_t):
        obs_t = self._sanitize(obs_t)
        logits = self.actor(obs_t)
        logits = self._sanitize(logits)
        return torch.distributions.Categorical(logits=logits), logits

    # ===== 动作相关 =====
    def _encode_action(self, next_hop):
        if next_hop not in self.action_map:
            if len(self.action_map) >= self.act_dim:
                idx = hash(next_hop) % self.act_dim
            else:
                idx = len(self.action_map)
            self.action_map[next_hop] = idx
            self.inverse_action_map[idx] = next_hop
        return self.action_map[next_hop]

    def _best_lq_next_hop(self, current):
        if not self.link_quality[current]:
            return ""
        return max(self.link_quality[current].items(), key=lambda kv: kv[1])[0]

    # ===== 公开接口（方法态）=====
    def update_lq(self, sf, ef, bf, current_node, next_hop):
        lq = self.a * sf + self.b * ef + self.c * bf
        self.link_quality[str(current_node)][str(next_hop)] = float(lq)
        return float(lq)

    def update_q_value(self, sf, ef, bf, current_node, next_hop, dest_node, reward):
        """
        on-policy 收集：把 (obs, gobs, act, logp, r, v, done) 推入缓冲；
        这里“done”无法由外部传入，暂按 next_hop==dest_node 近似为终止。
        返回值：当前 Critic 估计的 V 值。
        """
        current_node = str(current_node)
        next_hop = str(next_hop)
        dest_node = str(dest_node)

        self.update_lq(float(sf), float(ef), float(bf), current_node, next_hop)

        aidx = self._encode_action(next_hop)
        self.state_action_set[(current_node, dest_node)].add(aidx)

        obs = self._obs(current_node, dest_node)
        gobs = self._gobs()

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        gobs_t = torch.tensor(gobs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            try:
                dist, _ = self._dist(obs_t)
            except ValueError:
                self._reinit_actor_critic()
                dist, _ = self._dist(obs_t)
            val = self.critic(self._sanitize(gobs_t)).item()
            logp = dist.log_prob(torch.tensor([aidx], device=self.device)).item()

        done = 1.0 if (next_hop == dest_node) else 0.0
        self.buf.add(obs, gobs, aidx, logp, float(reward), float(val), float(done))

        self.update_counter += 1
        if len(self.buf) >= self.buf.capacity and (self.update_counter % self.replay_interval == 0):
            self._train()
            self.buf.clear()
            self.update_counter = 0
            self._save()
        return val

    @torch.no_grad()
    def get_best_next_hop(self, current_node, dest_node=None):
        current_node = str(current_node)
        if dest_node is None:
            return self._best_lq_next_hop(current_node) or ""

        dest_node = str(dest_node)
        candidates = list(self.state_action_set.get((current_node, dest_node), []))
        if not candidates:
            if self.link_quality[current_node]:
                for nh in self.link_quality[current_node].keys():
                    candidates.append(self._encode_action(nh))
            if not candidates:
                return ""

        obs = torch.tensor(self._obs(current_node, dest_node), dtype=torch.float32, device=self.device).unsqueeze(0)
        _, logits = self._dist(obs)
        logits = logits[0].cpu().numpy()
        best = max(candidates, key=lambda k: logits[k] if k < len(logits) else -1e9)
        return self.inverse_action_map.get(best, "")

    def _train(self):
        obs, gobs, act, oldp, adv, ret = self.buf.gae(last_v=0.0)

        N = obs.shape[0]
        idx = np.arange(N)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, N, self.mini_batch):
                e = min(s + self.mini_batch, N)
                mb = idx[s:e]

                mb_obs, mb_gobs = obs[mb], gobs[mb]
                mb_act, mb_oldp = act[mb], oldp[mb]
                mb_adv, mb_ret = adv[mb], ret[mb]

                dist, _ = self._dist(mb_obs)
                newp = dist.log_prob(mb_act)
                ratio = torch.exp(newp - mb_oldp)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy = dist.entropy().mean()

                v = self.critic(self._sanitize(mb_gobs))
                value_loss = F.mse_loss(v, mb_ret)

                loss = policy_loss - self.ent_coef * entropy + self.vf_coef * value_loss

                self.opt_pi.zero_grad()
                self.opt_v.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
                self.opt_pi.step()
                self.opt_v.step()

    # ===== 模型存储 =====
    def _ckpt(self):
        return "models/qoar_mappo.pth"

    def _save(self):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "opt_pi": self.opt_pi.state_dict(),
                "opt_v": self.opt_v.state_dict(),
                "node_map": self.node_map,
                "action_map": self.action_map,
                "inverse_action_map": self.inverse_action_map,
                "state_action_set": dict((k, list(v)) for k, v in self.state_action_set.items()),
                "a": self.a,
                "b": self.b,
                "c": self.c,
                "gamma": self.gamma,
                "pi_lr": self.pi_lr,
                "vf_lr": self.vf_lr,
            },
            self._ckpt(),
        )

    def _load(self):
        p = self._ckpt()
        if not os.path.exists(p):
            return
        try:
            ckpt = torch.load(p, map_location=self.device)
            self.actor.load_state_dict(ckpt["actor"])
            self.critic.load_state_dict(ckpt["critic"])
            self.opt_pi.load_state_dict(ckpt["opt_pi"])
            self.opt_v.load_state_dict(ckpt["opt_v"])
            self.node_map = ckpt.get("node_map", {})
            self.action_map = ckpt.get("action_map", {})
            self.inverse_action_map = ckpt.get("inverse_action_map", {})
            s = ckpt.get("state_action_set", {})
            self.state_action_set = defaultdict(set, {k: set(v) for k, v in s.items()})
            self.a = ckpt.get("a", self.a)
            self.b = ckpt.get("b", self.b)
            self.c = ckpt.get("c", self.c)
            self.gamma = ckpt.get("gamma", self.gamma)
            self.pi_lr = ckpt.get("pi_lr", self.pi_lr)
            self.vf_lr = ckpt.get("vf_lr", self.vf_lr)
            for pg in self.opt_pi.param_groups:
                pg["lr"] = self.pi_lr
            for pg in self.opt_v.param_groups:
                pg["lr"] = self.vf_lr
            if VERBOSE:
                print(f"[MAPPO] 模型已加载：{p}")
        except Exception as e:
            if VERBOSE:
                print(f"[MAPPO] 加载失败：{e}")


# ===== 全局单例（保持原名与函数）=====
qlearning = MAPPOQoAR()


# --------- 实用工具：弹性解析 ---------
def _is_num(x):
    return isinstance(x, numbers.Number) and not isinstance(x, bool)

def _normalize_update_args(*args, **kwargs):
    """
    统一还原为 (sf, ef, bf, current_node, next_hop, dest_node, reward)
    支持多种少参形态：
      - 7参：sf,ef,bf,current,next_hop,dest,reward
      - 6参：sf,ef,bf,current,dest,reward   -> 自动补 next_hop
      - 5参：sf,ef,bf,current,dest          -> 自动补 next_hop，reward=0
      - 4参：current,next_hop,dest,reward    或 sf,ef,bf,current
      - 3参：current,dest,reward            -> 自动补 next_hop
      - 2/1参：尽量兜底（训练信息有限）
    也可使用命名参数：sf/ef/bf/current_node/next_hop/dest_node/reward。
    """
    sf = kwargs.get("sf")
    ef = kwargs.get("ef")
    bf = kwargs.get("bf")
    current = kwargs.get("current_node")
    next_hop = kwargs.get("next_hop")
    dest = kwargs.get("dest_node")
    reward = kwargs.get("reward")

    n = len(args)

    def _ensure_next_by_policy(cur, dst):
        if cur is None:
            return None
        if dst is not None:
            nh = qlearning.get_best_next_hop(cur, dst)
            if nh:
                return nh
        nh = qlearning.get_best_next_hop(cur)
        return nh or ""

    if n == 7:
        sf, ef, bf, current, next_hop, dest, reward = args
    elif n == 6:
        a0, a1, a2, a3, a4, a5 = args
        if _is_num(a0) and _is_num(a1) and _is_num(a2):
            sf, ef, bf, current = a0, a1, a2, a3
            if _is_num(a4) and not _is_num(a5):
                reward, dest = float(a4), str(a5); next_hop = _ensure_next_by_policy(current, dest)
            elif _is_num(a5) and not _is_num(a4):
                reward, dest = float(a5), str(a4); next_hop = _ensure_next_by_policy(current, dest)
            else:
                next_hop, dest = str(a4), str(a5)
                reward = 0.0 if reward is None else float(reward)
        else:
            strs = [x for x in args if isinstance(x, str)]
            nums = [x for x in args if _is_num(x)]
            if len(strs) >= 3 and len(nums) >= 1:
                current, next_hop, dest = map(str, strs[:3])
                reward = float(nums[0])
                sf = 0.0 if sf is None else float(sf)
                ef = 0.0 if ef is None else float(ef)
                bf = 0.0 if bf is None else float(bf)
    elif n == 5:
        a0, a1, a2, a3, a4 = args
        if _is_num(a0) and _is_num(a1) and _is_num(a2):
            sf, ef, bf, current, dest = a0, a1, a2, a3, a4
            next_hop = _ensure_next_by_policy(current, dest)
            reward = 0.0 if reward is None else float(reward)
        else:
            strs = [x for x in args if isinstance(x, str)]
            nums = [x for x in args if _is_num(x)]
            if len(strs) >= 3 and len(nums) >= 1:
                current, next_hop, dest = map(str, strs[:3])
                reward = float(nums[0])
                sf = 0.0 if sf is None else float(sf)
                ef = 0.0 if ef is None else float(ef)
                bf = 0.0 if bf is None else float(bf)
    elif n == 4:
        a0, a1, a2, a3 = args
        if _is_num(a0) and _is_num(a1) and _is_num(a2) and isinstance(a3, str):
            sf, ef, bf, current = a0, a1, a2, a3
            dest = kwargs.get("dest_node", None)
            next_hop = kwargs.get("next_hop", _ensure_next_by_policy(current, dest))
            if dest is None:
                dest = next_hop or current
            reward = 0.0 if reward is None else float(reward)
        elif isinstance(a0, str) and isinstance(a1, str) and isinstance(a2, str) and _is_num(a3):
            current, next_hop, dest, reward = str(a0), str(a1), str(a2), float(a3)
            sf = 0.0 if sf is None else float(sf)
            ef = 0.0 if ef is None else float(ef)
            bf = 0.0 if bf is None else float(bf)
    elif n == 3:
        a0, a1, a2 = args
        if isinstance(a0, str) and isinstance(a1, str) and _is_num(a2):
            current, dest, reward = str(a0), str(a1), float(a2)
            next_hop = _ensure_next_by_policy(current, dest)
            sf = 0.0 if sf is None else float(sf)
            ef = 0.0 if ef is None else float(ef)
            bf = 0.0 if bf is None else float(bf)
    elif n == 2:
        a0, a1 = args
        if isinstance(a0, str) and isinstance(a1, str):
            current, dest = str(a0), str(a1)
            next_hop = _ensure_next_by_policy(current, dest)
            reward = 0.0 if reward is None else float(reward)
            sf = 0.0 if sf is None else float(sf)
            ef = 0.0 if ef is None else float(ef)
            bf = 0.0 if bf is None else float(bf)
    elif n == 1:
        a0 = args[0]
        if isinstance(a0, str):
            current = str(a0)
            next_hop = qlearning.get_best_next_hop(current) or ""
            dest = next_hop or current
            reward = 0.0 if reward is None else float(reward)
            sf = 0.0 if sf is None else float(sf)
            ef = 0.0 if ef is None else float(ef)
            bf = 0.0 if bf is None else float(bf)

    sf = 0.0 if sf is None else float(sf)
    ef = 0.0 if ef is None else float(ef)
    bf = 0.0 if bf is None else float(bf)
    current = "" if current is None else str(current)
    if next_hop is None or next_hop == "":
        next_hop = qlearning.get_best_next_hop(current, dest) if dest else qlearning.get_best_next_hop(current)
        next_hop = next_hop or ""
    if dest is None or dest == "":
        dest = next_hop or current
    reward = 0.0 if reward is None else float(reward)

    return sf, ef, bf, current, next_hop, dest, reward


# --------- 模块级函数（保持旧名），增加弹性 ----------
def update_q_value(*args, **kwargs):
    sf, ef, bf, current, next_hop, dest, reward = _normalize_update_args(*args, **kwargs)
    print((f"[QoAR] set params failed: {sf, ef, bf, current, next_hop, dest, reward}"))
    return qlearning.update_q_value(sf, ef, bf, current, next_hop, dest, reward)


def update_lq(sf=None, ef=None, bf=None, current_node=None, next_hop=None, *args, **kwargs):
    if sf is None or ef is None or bf is None:
        sf = 0.0 if sf is None else float(sf)
        ef = 0.0 if ef is None else float(ef)
        bf = 0.0 if bf is None else float(bf)
    if current_node is None:
        current_node = kwargs.get("current") or ""
    if next_hop is None:
        next_hop = kwargs.get("nh") or qlearning.get_best_next_hop(str(current_node)) or ""
    return qlearning.update_lq(float(sf), float(ef), float(bf), str(current_node), str(next_hop))


def get_best_next_hop(*args, **kwargs):
    if len(args) >= 2:
        return qlearning.get_best_next_hop(str(args[0]), str(args[1]))
    elif len(args) == 1:
        return qlearning.get_best_next_hop(str(args[0]))
    cur = kwargs.get("current_node") or kwargs.get("current") or ""
    dst = kwargs.get("dest_node") or kwargs.get("dest")
    if dst is None:
        return qlearning.get_best_next_hop(str(cur))
    return qlearning.get_best_next_hop(str(cur), str(dst))


# ===== 参数归一化/应用（静默）=====
def _normalize_abcs(a=None, b=None, c=None):
    if a is None: a = 0.5
    if b is None: b = 0.0
    if c is None: c = 1.0 - float(a) - float(b)
    a, b, c = float(a), float(b), float(c)
    a = max(a, 0.0); b = max(b, 0.0); c = max(c, 0.0)
    s = a + b + c
    if s <= 0.0:
        a, b, c = 0.5, 0.0, 0.5
    else:
        a, b, c = a / s, b / s, c / s
    return a, b, c

def _normalize_params(*args, **kwargs):
    alpha = kwargs.get("alpha", kwargs.get("pi_lr", 0.8))
    gamma = kwargs.get("gamma", kwargs.get("discount", 0.9))
    a = kwargs.get("a", None)
    b = kwargs.get("b", None)
    c = kwargs.get("c", None)

    if len(args) == 1:
        one = args[0]
        if isinstance(one, dict):
            lower = {str(k).lower(): v for k, v in one.items()}
            alpha = lower.get("alpha", lower.get("pi_lr", alpha))
            gamma = lower.get("gamma", lower.get("discount", gamma))
            a = lower.get("a", a); b = lower.get("b", b); c = lower.get("c", c)
        elif isinstance(one, (list, tuple)):
            seq = list(one)
            if len(seq) >= 1: alpha = seq[0]
            if len(seq) >= 2: gamma = seq[1]
            if len(seq) >= 3: a = seq[2]
            if len(seq) >= 4: b = seq[3]
            if len(seq) >= 5: c = seq[4]
    elif len(args) >= 2:
        alpha = args[0]
        gamma = args[1]
        if len(args) >= 3: a = args[2]
        if len(args) >= 4: b = args[3]
        if len(args) >= 5: c = args[4]

    a, b, c = _normalize_abcs(a, b, c)
    try:
        alpha = float(alpha)
    except Exception:
        alpha = 0.8
    try:
        gamma = float(gamma)
    except Exception:
        gamma = 0.9
    return float(alpha), float(gamma), float(a), float(b), float(c)

def _apply_params(alpha, gamma, a, b, c):
    try:
        qlearning.set_parameters(alpha, gamma, a, b, c)
        for pg in qlearning.opt_pi.param_groups:
            pg["lr"] = qlearning.pi_lr
        for pg in qlearning.opt_v.param_groups:
            pg["lr"] = qlearning.vf_lr
        if VERBOSE:
            print(f"[QoAR] params: alpha={alpha}, gamma={gamma}, a={a}, b={b}, c={c}")
        return True
    except Exception as e:
        if VERBOSE:
            print(f"[QoAR] set params failed: {e}")
        return False

def set_qlearning_params(*args, **kwargs):
    alpha, gamma, a, b, c = _normalize_params(*args, **kwargs)
    return _apply_params(alpha, gamma, a, b, c)

def set_mappo_params(*args, **kwargs):
    alpha, gamma, a, b, c = _normalize_params(*args, **kwargs)
    return _apply_params(alpha, gamma, a, b, c)

# 别名（避免反射名字不匹配）
def po_params(*args, **kwargs):
    return set_mappo_params(*args, **kwargs)

def set_po_params(*args, **kwargs):
    return set_mappo_params(*args, **kwargs)


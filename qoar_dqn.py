# === DDQN 版本的 QoAR (单文件替换 MAPPO 部分) ===
# 说明：保留外部接口名称（update_q_value, get_best_next_hop, update_lq, set_qlearning_params, set_mappo_params）
# 使用 Double DQN（在线 Q + 目标 Q），经验回放，action mask 支持。
import os
import numbers
import numpy as np
from collections import defaultdict, deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

# ========== 简单 MLP Q 网络 ==========
class QNet(nn.Module):
    def __init__(self, in_dim, hidden=(128,128), out_dim=64):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)   # 返回 Q-values（未 softmax）

# ========== 经验回放 ==========
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)
    def push(self, s, a, r, s2, done, mask, gobs, gobs2):
        # 保存：state(obs), action_idx, reward, next_state, done, action_mask, gobs, next_gobs
        self.buffer.append((s, a, r, s2, float(done), mask, gobs, gobs2))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done, mask, gobs, gobs2 = map(np.array, zip(*batch))
        return s, a.astype(np.int64), r.astype(np.float32), s2, done.astype(np.float32), mask, gobs, gobs2
    def __len__(self):
        return len(self.buffer)
    def clear(self):
        self.buffer.clear()

# ========== DDQN QoAR Agent (保留接口) ==========
class DDQNQoAR:
    def __init__(self, lr=0.001, gamma=0.99, buffer_size=20000, batch_size=256,
                 act_dim=64, H=64, target_update=1000, min_replay=1024,
                 eps_start=1.0, eps_end=0.05, eps_decay=20000, device=None):
        # 参数
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.act_dim = int(act_dim)
        self.H = int(H)
        self.obs_dim = self.H * 2 + 3
        self.gobs_dim = self.obs_dim

        # maps and sets (保留你的结构)
        self.action_map = {}            # key -> idx
        self.inverse_action_map = {}    # idx -> key
        self.state_action_set = defaultdict(set)  # (cur,dst) -> set(aidx)
        self.node_map = {}
        self.last_obs_by_node = defaultdict(lambda: np.zeros(self.obs_dim, dtype=np.float32))
        self.link_quality = defaultdict(lambda: defaultdict(float))

        # device
        if device is None:
            self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
            print(f"[DDQNQoAR] 使用设备: {self.device}")
        else:
            self.device = torch.device(device)

        # 网络：在线和目标
        self.q_online = QNet(self.obs_dim, out_dim=self.act_dim).to(self.device)
        self.q_target = QNet(self.obs_dim, out_dim=self.act_dim).to(self.device)
        self.q_target.load_state_dict(self.q_online.state_dict())

        self.opt = torch.optim.Adam(self.q_online.parameters(), lr=self.lr)

        # replay
        self.replay = ReplayBuffer(buffer_size)
        self.batch_size = int(batch_size)
        self.min_replay = int(min_replay)

        # epsilon-greedy
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay = int(eps_decay)
        self.total_steps = 0

        # target update
        self.target_update = int(target_update)
        self.learn_steps = 0

        # diagnostics storage (logs)
        self.rewards_log = []
        self.loss_log = []
        self.qvalue_log = []

        # save dir
        os.makedirs("models", exist_ok=True)
        self._load()
        # sanitize
        if not self._model_finite(self.q_online) or not self._model_finite(self.q_target):
            self._reinit_networks()
            try:
                os.remove(self._ckpt())
            except Exception:
                pass

    # ---------- utils ----------
    def _ckpt(self):
        return "models/qoar_ddqn.pth"

    def _save(self):
        torch.save({
            "q_online": self.q_online.state_dict(),
            "q_target": self.q_target.state_dict(),
            "opt": self.opt.state_dict(),
            "node_map": self.node_map,
            "action_map": self.action_map,
            "inverse_action_map": self.inverse_action_map,
            "state_action_set": {k: list(v) for k,v in self.state_action_set.items()},
            "gamma": self.gamma,
            "lr": self.lr,
            "eps": (self.eps_start, self.eps_end, self.eps_decay),
        }, self._ckpt())

    def _load(self):
        p = self._ckpt()
        if not os.path.exists(p):
            return
        try:
            ck = torch.load(p, map_location=self.device)
            self.q_online.load_state_dict(ck["q_online"])
            self.q_target.load_state_dict(ck["q_target"])
            self.opt.load_state_dict(ck["opt"])
            self.node_map = ck.get("node_map", {})
            self.action_map = ck.get("action_map", {})
            self.inverse_action_map = ck.get("inverse_action_map", {})
            s = ck.get("state_action_set", {})
            self.state_action_set = defaultdict(set, {k: set(v) for k, v in s.items()})
            self.gamma = ck.get("gamma", self.gamma)
            self.lr = ck.get("lr", self.lr)
            eps_tuple = ck.get("eps", None)
            if eps_tuple is not None:
                self.eps_start, self.eps_end, self.eps_decay = eps_tuple
            print(f"[DDQN] 模型已加载：{p}")
        except Exception as e:
            print(f"[DDQN] 加载失败：{e}")

    def _model_finite(self, model: nn.Module) -> bool:
        for p in model.parameters():
            if not torch.isfinite(p).all():
                return False
        return True

    def _reinit_networks(self):
        def init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.q_online.apply(init)
        self.q_target.apply(init)

    def plot_training_curves(self, save_path=None):
        if save_path:
           os.makedirs(save_path, exist_ok=True)
        smooth_window=1000
        # print(self.policy_loss_log)
        # print(self.value_loss_log)
        # print(self.loss_log)
        # print(f"[MAPPOQoAR] 绘制训练曲线，数据点数：奖励 {len(self.rewards_log)}，策略损失 {len(self.policy_loss_log)}，值损失 {len(self.value_loss_log)}, 总损失 {len(self.loss_log)}")
        plt.figure(figsize=(12, 5))
        # --- Reward 曲线 ---
        plt.title("Reward Curve")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        # plt.plot(self.rewards_log, color='tab:blue', alpha=0.3, label='Raw Reward')  # 原始奖励，透明显示
        if len(self.rewards_log) > smooth_window:
            smooth = np.convolve(self.rewards_log, np.ones(smooth_window)/smooth_window, mode='same')
            valid_len = len(self.rewards_log) - smooth_window // 2
            smooth = smooth[:valid_len]
            # plt.plot(range(valid_len), smooth, color='tab:orange', label=f'Smoothed ({smooth_window})')
            plt.plot(range(valid_len), smooth, color='tab:orange', label='Reward')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        if save_path:
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
            # filename = f"reward_curve_{timestamp}.png"
            filename = "reward_curve.png"
            plt.savefig(os.path.join(save_path, filename), dpi=300)
        plt.show()
    
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
            stats = np.array([0.0,0.0,0.0], dtype=np.float32)
        else:
            stats = np.array([float(np.max(lqs)), float(np.mean(lqs)), float(np.min(lqs))], dtype=np.float32)
        obs = np.concatenate([cur, dst, stats], axis=0)
        self.last_obs_by_node[current] = obs
        return obs

    def _gobs(self):
        if not self.last_obs_by_node:
            return np.zeros(self.gobs_dim, dtype=np.float32)
        agg = np.sum(np.stack(list(self.last_obs_by_node.values()), axis=0), axis=0)
        return np.clip(agg, 0.0, 1.0).astype(np.float32)

    def _build_action_mask(self, current_node):
        # returns tensor shape (act_dim,) of 0/1
        mask = np.zeros(self.act_dim, dtype=np.float32)
        # state_action_set stores action idx's per (current,dest) keys in your system;
        # but you might want mask based purely on current -> next_hop candidates.
        # We'll check state_action_set entries keyed by tuples where cur==current_node
        for (cur, dst), acts in self.state_action_set.items():
            if cur == current_node:
                for aidx in acts:
                    if isinstance(aidx, (int, np.integer)):
                        if 0 <= int(aidx) < self.act_dim:
                            mask[int(aidx)] = 1.0
        if mask.sum() == 0:
            mask[:] = 1.0
        return torch.tensor(mask, device=self.device)

    # ---------- action encoding (keep behavior) ----------
    def _encode_action(self, next_hop, band):
        key = (str(next_hop), int(band))
        if key not in self.action_map:
            if len(self.action_map) >= self.act_dim:
                idx = hash(key) % self.act_dim
            else:
                idx = len(self.action_map)
            self.action_map[key] = idx
            self.inverse_action_map[idx] = key
        return self.action_map[key]

    def _best_lq_next_hop(self, current):
        if not self.link_quality[current]:
            return ""
        return max(self.link_quality[current].items(), key=lambda kv: kv[1])[0]

    # ---------- public compatible interfaces ----------
    def update_lq(self, sf, df, bf, current_node, next_hop, band=0):
        lq = self.a * float(sf) + self.b * float(df) + self.c * float(bf)
        self.link_quality[str(current_node)][(str(next_hop), int(band))] = float(lq)
        return float(lq)

    def get_epsilon(self):
        # linear decay
        eps = self.eps_end + (self.eps_start - self.eps_end) * max(0.0, (1.0 - self.total_steps / float(self.eps_decay)))
        return float(eps)

    def select_action(self, obs, mask_tensor=None):
        # obs: numpy array single obs; mask_tensor: torch tensor (act_dim,)
        eps = self.get_epsilon()
        if random.random() < eps:
            # random among valid actions
            if mask_tensor is None:
                return int(random.randrange(self.act_dim))
            else:
                mask_np = mask_tensor.cpu().numpy()
                valid = np.nonzero(mask_np)[0]
                if len(valid) == 0:
                    return int(random.randrange(self.act_dim))
                return int(np.random.choice(valid))
        else:
            # greedy using online Q
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.q_online(obs_t)[0]  # [act_dim]
            qvals = torch.nan_to_num(qvals, nan=-1e8, posinf=1e8, neginf=-1e8)
            if mask_tensor is not None:
                # mask invalid by -inf
                masked = qvals.clone()
                masked = masked + (mask_tensor + 1e-8).log()  # mask 0 -> -inf
                chosen = int(torch.argmax(masked).item())
            else:
                chosen = int(torch.argmax(qvals).item())
            return chosen

    def get_best_next_hop(self, current_node, dest_node=None):
        # keep same signature; return (next_hop, band)
        current_node = str(current_node)
        # collect candidates as earlier
        candidates = []
        for (cur, dst), acts in self.state_action_set.items():
            if cur == current_node:
                candidates.extend(acts)
        candidates = [int(aid) for aid in set(candidates) if isinstance(aid, (int, np.integer))]
        if not candidates:
            if self.link_quality[current_node]:
                best_nh, best_band = max(self.link_quality[current_node].items(), key=lambda kv: kv[1])[0]
                return best_nh, best_band
            return "", 0

        obs = self.last_obs_by_node.get(current_node, np.zeros(self.obs_dim, dtype=np.float32))
        mask_tensor = self._build_action_mask(current_node)
        chosen_aidx = self.select_action(obs, mask_tensor)
        # ensure chosen in candidates; if not, force choose best among candidates by Q
        if chosen_aidx not in candidates:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.q_online(obs_t)[0].cpu().detach().numpy()
            valid_candidates = [a for a in candidates if 0 <= a < len(qvals)]
            if not valid_candidates:
                if self.link_quality[current_node]:
                    best_nh, best_band = max(self.link_quality[current_node].items(), key=lambda kv: kv[1])[0]
                    return best_nh, best_band
                return "", 0
            chosen_aidx = int(max(valid_candidates, key=lambda k: qvals[k]))
        nxt = self.inverse_action_map.get(chosen_aidx, ("", 0))
        return str(nxt[0]), int(nxt[1])

    def update_q_value(self, sf, df, bf, current_node, next_hop, dest_node, band, reward):
        """
        For compatibility: called every step by ns3 wrapper.
        We'll:
         - update link_quality
         - encode action, add to state_action_set
         - push transition into replay buffer (using last obs & new obs)
         - optionally trigger training when replay size >= min_replay
        Returns current Q estimate for state-action (online net)
        """
        current_node = str(current_node)
        next_hop = str(next_hop)
        dest_node = str(dest_node)

        # update lq table (use stored a/b/c if exist; ensure they exist)
        try:
            a = self.a
            b = self.b
            c = self.c
        except Exception:
            # fallback defaults
            a,b,c = 0.4, 0.2, 0.4
            self.a, self.b, self.c = a,b,c
        self.update_lq(float(sf), float(df), float(bf), current_node, next_hop, band)

        aidx = self._encode_action(next_hop, band)
        self.state_action_set[(current_node, dest_node)].add(aidx)

        # build obs and next_obs (we don't have environment step notion; approximate next_obs by last_obs_by_node after update)
        obs = self._obs(current_node, dest_node)
        gobs = self._gobs()

        # For next state we can use same since ns3 calls update_q_value per observation;
        # but to make replay meaningful, treat next_obs as current last_obs (no env step)
        # If desired, external should call with next state's values (compat).
        next_obs = obs.copy()
        next_gobs = gobs.copy()

        # compute Q estimate for (obs, aidx)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qvals = self.q_online(obs_t)[0]
            qvals = torch.nan_to_num(qvals, nan=0.0, posinf=1e8, neginf=-1e8)
            q_est = float(qvals[aidx].item())

        # push to replay buffer
        # build action mask for current node
        mask = self._build_action_mask(self._idx(current_node)).cpu().numpy() if isinstance(self._idx(current_node), int) else np.ones(self.act_dim)
        self.replay.push(obs, aidx, float(reward), next_obs, (next_hop == dest_node), mask, gobs, next_gobs)
        self.rewards_log.append(float(reward))
        self.total_steps += 1

        # learning trigger
        if len(self.replay) >= self.min_replay:
            # train for a few gradient steps each call (configurable)
            self._learn(steps=1)

        # target update by steps
        if self.learn_steps > 0 and (self.learn_steps % self.target_update == 0):
            self.q_target.load_state_dict(self.q_online.state_dict())

        # occasionally save
        if self.total_steps % 10000 == 0:
            try:
                self._save()
            except Exception:
                pass

        return q_est

    # ---------- core DDQN update ----------
    def _learn(self, steps=1):
        if len(self.replay) < self.batch_size:
            return
        for _ in range(steps):
            s, a, r, s2, done, mask, gobs, gobs2 = self.replay.sample(self.batch_size)
            # convert to tensors
            s_t = torch.tensor(s, dtype=torch.float32, device=self.device)
            a_t = torch.tensor(a, dtype=torch.long, device=self.device).unsqueeze(1)
            r_t = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
            s2_t = torch.tensor(s2, dtype=torch.float32, device=self.device)
            done_t = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

            # Q online for current states
            q_vals = self.q_online(s_t).gather(1, a_t)  # [B,1]

            # Double DQN target:
            # 1) actions from online network (argmax)
            with torch.no_grad():
                q_online_next = self.q_online(s2_t)  # [B, act_dim]
                q_target_next = self.q_target(s2_t)  # [B, act_dim]

                # apply mask if provided (mask array shape (B, act_dim))
                if mask is not None:
                    mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device)
                    # mask_t 0 -> -inf
                    q_online_next = q_online_next + (mask_t + 1e-8).log()
                    q_target_next = q_target_next + (mask_t + 1e-8).log()

                next_actions = torch.argmax(q_online_next, dim=1, keepdim=True)  # [B,1]
                next_q_target = q_target_next.gather(1, next_actions)  # [B,1]

                target = r_t + (1.0 - done_t) * (self.gamma * next_q_target)

            # loss (MSE)
            loss = F.mse_loss(q_vals, target)
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_online.parameters(), 0.5)
            self.opt.step()

            self.learn_steps += 1
            self.loss_log.append(float(loss.item()))
            # log q value mean
            with torch.no_grad():
                self.qvalue_log.append(float(q_vals.mean().item()))

    # parameter setting compatible functions
    def set_parameters(self, alpha=None, gamma=None, a=None, b=None, c=None):
        # keep compat: alpha -> lr
        if alpha is not None:
            try:
                self.lr = float(alpha)
                for pg in self.opt.param_groups:
                    pg['lr'] = self.lr
            except Exception:
                pass
        if gamma is not None:
            self.gamma = float(gamma)
        if a is not None and b is not None and c is not None:
            if abs(a+b+c - 1.0) > 1e-6:
                raise ValueError("a+b+c must sum to 1")
            self.a, self.b, self.c = float(a), float(b), float(c)

    # external parameter API compatibility wrappers
    def set_qlearning_params(self, *args, **kwargs):
        alpha, gamma, a, b, c = _normalize_params(*args, **kwargs)
        self.set_parameters(alpha, gamma, a, b, c)
        return True

    def set_mappo_params(self, *args, **kwargs):
        return self.set_qlearning_params(*args, **kwargs)

# instantiate global single instance (keep name qlearning)
qlearning = DDQNQoAR()

# keep wrapper functions compatible at module level
def update_q_value(*args, **kwargs):
    sf, df, bf, current, next_hop, dest, band, reward = _normalize_update_args(*args, **kwargs)
    return qlearning.update_q_value(sf, df, bf, current, next_hop, dest, band, reward)

def update_lq(sf=None, ef=None, bf=None, current_node=None, next_hop=None, band=0, **kwargs):
    if sf is None or ef is None or bf is None:
        sf = 0.0 if sf is None else float(sf)
        ef = 0.0 if ef is None else float(ef)
        bf = 0.0 if bf is None else float(bf)
    if current_node is None:
        current_node = kwargs.get("current") or ""
    if next_hop is None:
        next_hop = kwargs.get("nh") or qlearning.get_best_next_hop(str(current_node)) or ""
    return qlearning.update_lq(float(sf), float(ef), float(bf), str(current_node), str(next_hop), band)

def _normalize_update_args(*args, **kwargs):
    """
    统一还原为 (sf, df, bf, current_node, next_hop, dest_node, reward)
    """
    # sf = kwargs.get("sf"); 
    # ef = kwargs.get("ef"); 
    # bf = kwargs.get("bf")
    # current = kwargs.get("current_node")
    # next_hop = kwargs.get("next_hop")
    # dest = kwargs.get("dest_node")
    # reward = kwargs.get("reward")
    # done = kwargs.get("done")

    n = len(args)

    if n == 9:
        sf, df, bf, current, next_hop, dest,band,reward, done = args

    # dou di ce lue
    sf = 0.0 if sf is None else float(sf)
    df = 0.0 if df is None else float(df)
    bf = 0.0 if bf is None else float(bf)
    current = "" if current is None else str(current)
    if next_hop is None or next_hop == "":
        next_hop = qlearning.get_best_next_hop(current, dest) if dest else qlearning.get_best_next_hop(current)
        next_hop = next_hop or ""
    if dest is None or dest == "":
        dest = next_hop or current
    reward = 0.0 if reward is None else float(reward)

    return sf, df, bf, current, next_hop, dest,band,reward
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

def _normalize_abcs(a=None, b=None, c=None):
    if a is None: a = 0.4
    if b is None: b = 0.2
    if c is None: c = 1.0 - float(a) - float(b)
    a, b, c = float(a), float(b), float(c)
    a = max(a, 0.0); b = max(b, 0.0); c = max(c, 0.0)
    s = a + b + c
    if s <= 0.0:
        a, b, c = 0.4, 0.2, 0.4
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

def set_qlearning_params(*args, **kwargs):
    return qlearning.set_qlearning_params(*args, **kwargs)

def set_mappo_params(*args, **kwargs):
    return qlearning.set_mappo_params(*args, **kwargs)

def plot_training_curves(save_path=None):
    qlearning.plot_training_curves(save_path=save_path)



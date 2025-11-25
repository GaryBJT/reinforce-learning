# MAPPO 版 QoAR（保持原 Q-learning 外部接口不变）
# - NaN/Inf 防护 + 坏权重自愈（Xavier重置+删除坏ckpt）
# - update_q_value / get_best_next_hop / update_lq 弹性少参兼容
# - set_mappo_params / set_qlearning_params 支持缺参(含缺 c)且静默
# ppo_lstm_agent.py
import os
import numbers
import numpy as np
from collections import defaultdict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical
from datetime import datetime

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
        # x can be [B, dim] or [B*T, dim] depending on caller
        x = x.view(-1, x.size(-1))
        for l in self.layers:
            x = F.relu(l(x))
        out = self.out(x) if self.out is not None else x
        return out


# ====== Actor / Critic with LSTM ======
class ActorLSTM(nn.Module):
    """
    --- REPLACED / NEW ---
    支持单步输入 [B, obs_dim] 或序列输入 [B, T, obs_dim] 并返回 logits 与新的 hidden
    """
    def __init__(self, obs_dim, act_dim, hidden=(128,), lstm_hidden=128, lstm_layers=1):
        super().__init__()
        self.encoder = MLP(obs_dim, hidden, out_dim=lstm_hidden)
        self.lstm = nn.LSTM(lstm_hidden, lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.policy = nn.Linear(lstm_hidden, act_dim)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, hx=None):
        """
        Supports:
          x: [B, obs_dim]  (single-step)
          x: [B, T, obs_dim] (sequence)
        Returns:
          if single-step:
            logits: [B, act_dim], (hn, cn)
          if sequence:
            logits: [B, T, act_dim], (hn, cn)
        """
        single = False
        if x.dim() == 2:
            single = True
            B = x.size(0)
            feat = self.encoder(x)  # [B, feat]
            feat = feat.unsqueeze(1)  # [B,1,feat]
        elif x.dim() == 3:
            B, T, D = x.shape
            feat = self.encoder(x.view(B * T, D))  # [B*T, feat]
            feat = feat.view(B, T, -1)
        else:
            raise ValueError("ActorLSTM.forward: x must be 2D or 3D")

        if hx is None:
            device = x.device
            num_layers = self.lstm.num_layers
            hidden_size = self.lstm.hidden_size
            h0 = torch.zeros(num_layers, feat.size(0), hidden_size, device=device)
            c0 = torch.zeros(num_layers, feat.size(0), hidden_size, device=device)
            hx = (h0, c0)

        out, (hn, cn) = self.lstm(feat, hx)  # out: [B, T, hidden] or [B,1,hidden]
        if single:
            out = out.squeeze(1)  # [B, hidden]
            logits = self.policy(out)  # [B, act_dim]
            return logits, (hn, cn)
        else:
            logits = self.policy(out)  # [B, T, act_dim]
            return logits, (hn, cn)


class CriticLSTM(nn.Module):
    """
    --- REPLACED / NEW ---
    支持单步或序列输入，输出相应形状 value
    """
    def __init__(self, gobs_dim, hidden=(128,), lstm_hidden=128, lstm_layers=1):
        super().__init__()
        self.encoder = MLP(gobs_dim, hidden, out_dim=lstm_hidden)
        self.lstm = nn.LSTM(lstm_hidden, lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.value_head = nn.Linear(lstm_hidden, 1)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, hx=None):
        """
        x: [B, gobs_dim] or [B, T, gobs_dim]
        returns:
          single-step: val [B], (hn,cn)
          sequence: val [B, T], (hn,cn)
        """
        single = False
        if x.dim() == 2:
            single = True
            B = x.size(0)
            feat = self.encoder(x).unsqueeze(1)  # [B,1,feat]
        elif x.dim() == 3:
            B, T, D = x.shape
            feat = self.encoder(x.view(B * T, D)).view(B, T, -1)
        else:
            raise ValueError("CriticLSTM.forward: x must be 2D or 3D")

        if hx is None:
            device = x.device
            num_layers = self.lstm.num_layers
            hidden_size = self.lstm.hidden_size
            h0 = torch.zeros(num_layers, feat.size(0), hidden_size, device=device)
            c0 = torch.zeros(num_layers, feat.size(0), hidden_size, device=device)
            hx = (h0, c0)

        out, (hn, cn) = self.lstm(feat, hx)
        if single:
            out = out.squeeze(1)
            val = self.value_head(out).squeeze(-1)
            return val, (hn, cn)
        else:
            val = self.value_head(out).squeeze(-1)  # [B, T]
            return val, (hn, cn)


# ====== On-policy 缓冲（GAE），增加隐状态存储 ======
class OnPolicyBuf:
    """
    --- REPLACED / NEW ---
    Buffer 能返回按序列组织的数据，并返回每序列的初始 hidden 列表（可能包含 None）
    """
    def __init__(self, capacity, gamma=0.95, lam=0.95, device=None):
        self.capacity = int(capacity)
        self.gamma = float(gamma)
        self.lam = float(lam)
        if device is None:
            self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.clear()

    def clear(self):
        self.obs, self.gobs, self.act = [], [], []
        self.old_logp, self.rew, self.val, self.done = [], [], [], []
        # 保存 actor/critic 隐状态（每步的 initial hidden）
        self.actor_hx, self.actor_cx = [], []
        self.critic_hx, self.critic_cx = [], []

    def add(self, obs, gobs, act, logp, rew, val, done, actor_h=None, actor_c=None, critic_h=None, critic_c=None):
        self.obs.append(obs)
        self.gobs.append(gobs)
        self.act.append(act)
        self.old_logp.append(logp)
        self.rew.append(rew)
        self.val.append(val)
        self.done.append(done)

        # actor_h, actor_c, critic_h, critic_c may be tensors with shape [num_layers, 1, hidden]
        # store as squeezed [num_layers, hidden] or None
        if actor_h is None:
            self.actor_hx.append(None)
            self.actor_cx.append(None)
        else:
            # ensure tensor and squeeze batch dim
            ah = actor_h.detach().cpu().squeeze(1) if isinstance(actor_h, torch.Tensor) else torch.tensor(actor_h)
            ac = actor_c.detach().cpu().squeeze(1) if isinstance(actor_c, torch.Tensor) else torch.tensor(actor_c)
            self.actor_hx.append(ah)
            self.actor_cx.append(ac)

        if critic_h is None:
            self.critic_hx.append(None)
            self.critic_cx.append(None)
        else:
            ch = critic_h.detach().cpu().squeeze(1) if isinstance(critic_h, torch.Tensor) else torch.tensor(critic_h)
            cc = critic_c.detach().cpu().squeeze(1) if isinstance(critic_c, torch.Tensor) else torch.tensor(critic_c)
            self.critic_hx.append(ch)
            self.critic_cx.append(cc)

    def __len__(self):
        return len(self.rew)

    def gae(self, last_v=0.0, seq_len=16):
        """
        --- REPLACED / NEW ---
        Compute GAE as before, then reshape returned tensors into sequences of shape [num_seq, seq_len, ...]
        Also return lists of stored initial hidden states for each sequence (may contain None).
        """
        if len(self.rew) == 0:
            raise ValueError("Buffer empty")

        r = torch.tensor(self.rew, dtype=torch.float32, device=self.device)
        v = torch.tensor(self.val, dtype=torch.float32, device=self.device)
        d = torch.tensor(self.done, dtype=torch.float32, device=self.device)

        N = len(self.rew)
        adv = torch.zeros_like(r)
        v_ext = torch.cat([v, torch.tensor([last_v], device=self.device)])
        gae = 0.0
        for t in reversed(range(N)):
            delta = r[t] + self.gamma * v_ext[t + 1] * (1 - d[t]) - v[t]
            gae = delta + self.gamma * self.lam * (1 - d[t]) * gae
            adv[t] = gae
        ret = adv + v

        # convert lists to arrays
        obs_arr = np.array(self.obs, dtype=np.float32)  # [N, obs_dim]
        gobs_arr = np.array(self.gobs, dtype=np.float32)
        act_arr = np.array(self.act, dtype=np.int64)
        oldp_arr = np.array(self.old_logp, dtype=np.float32)

        L = int(seq_len)
        num_seq = N // L
        if num_seq == 0:
            raise ValueError(f"Not enough data for one sequence of length {L}. N={N}")

        used = num_seq * L
        # reshape
        obs_seq = torch.tensor(obs_arr[:used].reshape(num_seq, L, -1), dtype=torch.float32, device=self.device)
        gobs_seq = torch.tensor(gobs_arr[:used].reshape(num_seq, L, -1), dtype=torch.float32, device=self.device)
        act_seq = torch.tensor(act_arr[:used].reshape(num_seq, L), dtype=torch.long, device=self.device)
        oldp_seq = torch.tensor(oldp_arr[:used].reshape(num_seq, L), dtype=torch.float32, device=self.device)
        adv_seq = adv[:used].view(num_seq, L).to(self.device)
        ret_seq = ret[:used].view(num_seq, L).to(self.device)
        v_seq = v[:used].view(num_seq, L).to(self.device)

        # initial hidden per sequence (choose saved hidden at first timestep of sequence)
        actor_h0 = []
        actor_c0 = []
        critic_h0 = []
        critic_c0 = []
        for s in range(num_seq):
            idx0 = s * L
            ah = self.actor_hx[idx0]
            ac = self.actor_cx[idx0]
            ch = self.critic_hx[idx0]
            cc = self.critic_cx[idx0]
            actor_h0.append(ah)
            actor_c0.append(ac)
            critic_h0.append(ch)
            critic_c0.append(cc)

        return obs_seq, gobs_seq, act_seq, oldp_seq, adv_seq, ret_seq, v_seq, actor_h0, actor_c0, critic_h0, critic_c0


# ====== 主体：PPO-LSTM 多智能体版（每节点实例化） ======
class MAPPOQoAR:
    """
    --- REPLACED / NEW ---
    序列 BPTT 版 PPO-LSTM agent (每节点一份实例化)
    ActorLSTM/CriticLSTM 支持序列输入，OnPolicyBuf 返回序列 + 初始 hidden
    """
    def __init__(self, alpha=0.01, gamma=0.9, a=0.4, b=0.2, c=0.4, buffer_size=100, batch_size=50, replay_interval=10, seq_len=16):
        # set params (keeps original style)
        self.set_parameters(alpha, gamma, a, b, c)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:1" if use_cuda else "cpu")
        print(f"[MAPPOQoAR] 使用设备: {self.device}")

        # obs dim
        self.H = 64
        self.obs_dim = self.H * 2 + 3
        self.gobs_dim = self.obs_dim

        # action space
        self.act_dim = 64
        self.action_map = {}
        self.inverse_action_map = {}
        self.state_action_set = defaultdict(set)

        self.node_map = {}
        self.last_obs_by_node = defaultdict(lambda: np.zeros(self.H * 2 + 3, dtype=np.float32))
        self.link_quality = defaultdict(lambda: defaultdict(float))

        # RNN hyperparams
        self.lstm_hidden = 128
        self.lstm_layers = 1
        self.seq_len = int(seq_len)  # --- NEW: sequence length for BPTT

        # actor / critic (LSTM)
        self.actor = ActorLSTM(self.obs_dim, self.act_dim, hidden=(128,), lstm_hidden=self.lstm_hidden, lstm_layers=self.lstm_layers).to(self.device)
        self.critic = CriticLSTM(self.gobs_dim, hidden=(128,), lstm_hidden=self.lstm_hidden, lstm_layers=self.lstm_layers).to(self.device)

        # 保存每个节点当前的 actor/critic 隐状态 (h,c)，格式: (h,c) each [num_layers, 1, hidden]
        self.rnn_states_actor = {}   # node_name -> (h,c)
        self.rnn_states_critic = {}  # node_name -> (h,c)

        # 优化器
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=self.pi_lr)
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=self.vf_lr)

        # PPO / GAE 超参
        self.clip_eps = 0.2
        self.lam = 0.95
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.epochs = 4
        self.mini_batch = 32  # minibatch measured in sequences

        # buffer
        self.train_batch = 2048
        self.buf = OnPolicyBuf(self.train_batch, gamma=self.gamma, lam=self.lam, device=self.device)
        self.update_counter = 0
        self.replay_interval = int(replay_interval)

        self.rewards_log = []
        self.policy_loss_log = []
        self.value_loss_log = []
        self.loss_log = []
        os.makedirs("models", exist_ok=True)
        # optionally load if exists
        self._load()
        if not self._model_finite(self.actor) or not self._model_finite(self.critic):
                print("[MAPPO] 检测到NaN/Inf权重，已重置并删除坏ckpt")
                self._reinit_actor_critic()
                # 清空优化器状态（防止旧梯度影响新参数）
                self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=self.pi_lr)
                self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=self.vf_lr)
        
                # 清空旧经验（如果有）
                if hasattr(self, "buf"):
                    self.buf.clear()
                    print("[MAPPO] 经验缓冲已清空")
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

    def _build_action_mask(self, current_node):
        mask = np.zeros(self.act_dim, dtype=np.float32)
        if current_node in self.state_action_set:
            for nh in self.state_action_set[current_node]:
                if nh in self.action_map:
                    idx = self.action_map[nh]
                    mask[idx] = 1.0
        if mask.sum() == 0:
            mask[:] = 1.0
        return torch.tensor(mask, dtype=torch.float32, device=self.device)

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

    # ====== RNN 隐状态工具 ======
    def _get_node_rnn_state(self, node_name):
        an = str(node_name)
        if an in self.rnn_states_actor:
            actor_h, actor_c = self.rnn_states_actor[an]
        else:
            num_layers = self.actor.lstm.num_layers
            hidden_size = self.actor.lstm.hidden_size
            device = self.device
            actor_h = torch.zeros(num_layers, 1, hidden_size, device=device)
            actor_c = torch.zeros(num_layers, 1, hidden_size, device=device)
            self.rnn_states_actor[an] = (actor_h, actor_c)
        if an in self.rnn_states_critic:
            critic_h, critic_c = self.rnn_states_critic[an]
        else:
            num_layers = self.critic.lstm.num_layers
            hidden_size = self.critic.lstm.hidden_size
            device = self.device
            critic_h = torch.zeros(num_layers, 1, hidden_size, device=device)
            critic_c = torch.zeros(num_layers, 1, hidden_size, device=device)
            self.rnn_states_critic[an] = (critic_h, critic_c)
        return (actor_h, actor_c), (critic_h, critic_c)

    def _set_node_actor_state(self, node_name, h, c):
        self.rnn_states_actor[str(node_name)] = (h.detach(), c.detach())

    def _set_node_critic_state(self, node_name, h, c):
        self.rnn_states_critic[str(node_name)] = (h.detach(), c.detach())

    # ====== 动作 / 环境更新方法 ======
    def update_lq(self, sf, df, bf, current_node, next_hop, band):
        lq = self.a * sf + self.b * df + self.c * bf
        self.link_quality[str(current_node)][(str(next_hop), band)] = float(lq)
        return float(lq)

    def update_q_value(self, sf, df, bf, current_node, next_hop, dest_node, band, reward):
        current_node = str(current_node)
        next_hop = str(next_hop)
        dest_node = str(dest_node)

        self.update_lq(float(sf), float(df), float(bf), current_node, next_hop, band)

        aidx = self._encode_action(next_hop, band)
        self.state_action_set[(current_node, dest_node)].add((aidx))

        obs = self._obs(current_node, dest_node)
        gobs = self._gobs()

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, obs_dim]
        gobs_t = torch.tensor(gobs, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 获取 rnn 隐状态
        (actor_h, actor_c), (critic_h, critic_c) = self._get_node_rnn_state(current_node)

        with torch.no_grad():
            mask = self._build_action_mask(int(obs[0]))
            logits, new_actor_state = self.actor(obs_t, hx=(actor_h, actor_c))
            mask_eps = (mask + 1e-8)
            masked_logits = logits + mask_eps.log().unsqueeze(0)
            masked_logits = self._sanitize(masked_logits)
            masked_logits = torch.clamp(masked_logits, -20, 20)
            dist = Categorical(logits=masked_logits)

            val, new_critic_state = self.critic(gobs_t, hx=(critic_h, critic_c))

            logp = dist.log_prob(torch.tensor([aidx], device=self.device)).item()

        done = 1.0 if (next_hop == dest_node) else 0.0

        # 保存初始 hidden（squeezed）到 buffer
        self.buf.add(obs, gobs, aidx, logp, float(reward), float(val.item()), float(done),
                     actor_h=actor_h.clone(), actor_c=actor_c.clone(),
                     critic_h=critic_h.clone(), critic_c=critic_c.clone())

        # 写回执行阶段新隐状态（保持在线记忆）
        self._set_node_actor_state(current_node, new_actor_state[0], new_actor_state[1])
        self._set_node_critic_state(current_node, new_critic_state[0], new_critic_state[1])

        self.update_counter += 1
        self.rewards_log.append(float(reward))

        if len(self.buf) >= self.buf.capacity:
            self._train()
            self.buf.clear()
            self.update_counter = 0
            self._save()

        return float(val.item())

    @torch.no_grad()
    def get_best_next_hop(self, current_node, dest_node=None):
        current_node = str(current_node)
        candidates = []
        for (cur, dst), acts in self.state_action_set.items():
            if cur == current_node:
                candidates.extend(acts)
        candidates = [int(c) for c in set(candidates) if isinstance(c, (int, np.integer))]
        if not candidates:
            if self.link_quality[current_node]:
                best_nh, band = max(self.link_quality[current_node].items(), key=lambda kv: kv[1])[0]
                return best_nh, band
            return "", 0

        obs = self.last_obs_by_node.get(current_node, np.zeros(self.obs_dim, dtype=np.float32))
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        (actor_h, actor_c), _ = self._get_node_rnn_state(current_node)
        logits, new_actor_state = self.actor(obs_t, hx=(actor_h, actor_c))
        logits = torch.nan_to_num(logits.squeeze(0), nan=0.0, posinf=1e6, neginf=-1e6)
        mask = self._build_action_mask(int(obs[0]))
        masked_logits = logits + (mask + 1e-8).log()
        probs = torch.softmax(masked_logits, dim=-1).cpu().numpy()

        valid_candidates = [a for a in candidates if 0 <= a < len(probs)]
        if not valid_candidates:
            if self.link_quality[current_node]:
                best_nh, band = max(self.link_quality[current_node].items(), key=lambda kv: kv[1])[0]
                return best_nh, band
            return "", 0

        cand_probs = np.array([probs[a] for a in valid_candidates], dtype=float)
        s = cand_probs.sum()
        if s > 1e-12:
            cand_probs /= s
            chosen_aidx = int(np.random.choice(valid_candidates, p=cand_probs))
        else:
            chosen_aidx = int(max(valid_candidates, key=lambda k: masked_logits[k].item()))

        next_hop, band = self.inverse_action_map.get(chosen_aidx, ("", 0))
        self._set_node_actor_state(current_node, new_actor_state[0], new_actor_state[1])
        return str(next_hop), int(band)

    def _train(self):
        """
        --- REPLACED / NEW ---
        使用序列 BPTT 训练：从 buf.gae() 获得 [num_seq, T, ...] 形式的数据
        """
        seq_len = self.seq_len
        # 得到序列化的数据与初始 hidden 列表
        obs_seq, gobs_seq, act_seq, oldp_seq, adv_seq, ret_seq, v_seq, \
            actor_h0_list, actor_c0_list, critic_h0_list, critic_c0_list = self.buf.gae(last_v=0.0, seq_len=seq_len)

        num_seq = obs_seq.shape[0]
        idxs = np.arange(num_seq)
        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            for start in range(0, num_seq, self.mini_batch):
                end = start + self.mini_batch
                mb_idx = idxs[start:end]
                if len(mb_idx) == 0:
                    continue

                mb_obs = obs_seq[mb_idx].to(self.device)     # [B, T, obs_dim]
                mb_gobs = gobs_seq[mb_idx].to(self.device)   # [B, T, gobs_dim]
                mb_act = act_seq[mb_idx].to(self.device)     # [B, T]
                mb_oldp = oldp_seq[mb_idx].to(self.device)   # [B, T]
                mb_adv = adv_seq[mb_idx].to(self.device)     # [B, T]
                mb_ret = ret_seq[mb_idx].to(self.device)     # [B, T]

                B = mb_obs.size(0)
                T = mb_obs.size(1)

                # build initial hidden states for minibatch
                num_layers = self.actor.lstm.num_layers
                hidden_size = self.actor.lstm.hidden_size
                device = self.device

                actor_h0 = torch.zeros(num_layers, B, hidden_size, device=device)
                actor_c0 = torch.zeros(num_layers, B, hidden_size, device=device)
                critic_h0 = torch.zeros(self.critic.lstm.num_layers, B, self.critic.lstm.hidden_size, device=device)
                critic_c0 = torch.zeros(self.critic.lstm.num_layers, B, self.critic.lstm.hidden_size, device=device)

                # fill from stored lists if present
                for i, seq_i in enumerate(mb_idx):
                    ah = actor_h0_list[seq_i]
                    ac = actor_c0_list[seq_i]
                    ch = critic_h0_list[seq_i]
                    cc = critic_c0_list[seq_i]
                    if ah is not None:
                        actor_h0[:, i, :] = ah.to(device)
                    if ac is not None:
                        actor_c0[:, i, :] = ac.to(device)
                    if ch is not None:
                        critic_h0[:, i, :] = ch.to(device)
                    if cc is not None:
                        critic_c0[:, i, :] = cc.to(device)

                # actor forward over sequence -> logits [B, T, A]
                logits_seq, _ = self.actor(mb_obs, hx=(actor_h0, actor_c0))
                # build masks per-step -> [B, T, A]
                masks = []
                for seq_row in mb_obs:  # seq_row shape [T, obs_dim]
                    mask_steps = []
                    for obs_step in seq_row:
                        current_node = int(obs_step[0].item())
                        mask_steps.append(self._build_action_mask(current_node))
                    masks.append(torch.stack(mask_steps, dim=0))
                masks = torch.stack(masks, dim=0).to(device)  # [B, T, A]
                masked_logits = logits_seq + (masks + 1e-8).log()

                # flatten for Categorical
                B, T, A = masked_logits.shape
                flat_logits = masked_logits.view(B * T, A)
                flat_actions = mb_act.view(B * T)
                dist = Categorical(logits=self._sanitize(flat_logits))

                newp = dist.log_prob(flat_actions)  # [B*T]
                newp = torch.nan_to_num(newp, nan=0.0)
                flat_oldp = mb_oldp.view(B * T)
                flat_adv = mb_adv.view(B * T)
                flat_ret = mb_ret.view(B * T)

                # PPO ratio
                logratio = newp - flat_oldp
                ratio = torch.exp(logratio).clamp(min=1e-8, max=1e8)

                surr1 = ratio * flat_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * flat_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                policy_loss = torch.clamp(policy_loss, min=-3.0, max=3.0)

                entropy = dist.entropy().mean()

                # critic forward sequence -> [B, T]
                v_seq_pred, _ = self.critic(mb_gobs, hx=(critic_h0, critic_c0))
                flat_v = v_seq_pred.contiguous().view(B * T)
                value_loss = F.mse_loss(flat_v, flat_ret)

                loss = policy_loss - self.ent_coef * entropy + self.vf_coef * value_loss

                # backward (BPTT)
                self.opt_pi.zero_grad()
                self.opt_v.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
                self.opt_pi.step()
                self.opt_v.step()

                # logging
                self.policy_loss_log.append(policy_loss.item())
                self.value_loss_log.append(value_loss.item())
                self.loss_log.append(loss.item())

        # adjust lr if needed
        self.adjust_lr()

    def _build_action_mask(self, current_node):
        mask = np.zeros(self.act_dim, dtype=np.float32)
        if current_node in self.state_action_set:
            for nh in self.state_action_set[current_node]:
                if nh in self.action_map:
                    idx = self.action_map[nh]
                    mask[idx] = 1.0
        if mask.sum() == 0:
            mask[:] = 1.0
        return torch.tensor(mask, device=self.device)

    def _ckpt(self):
        return "models/qoar_lstm.pth"

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
            print(f"无模型文件：{p}")
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

            print(f"[MAPPO] 模型已加载：{p}")
        except Exception as e:
            print(f"[MAPPO] 加载失败：{e}")

    def adjust_lr(self):
        decay_rate = 0.95
        new_lr = self.pi_lr * decay_rate
        self.pi_lr = max(1e-4, new_lr)
        self.vf_lr = max(1e-4, new_lr)
        for pg in self.opt_pi.param_groups:
            pg["lr"] = self.pi_lr
        for pg in self.opt_v.param_groups:
            pg["lr"] = self.vf_lr


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
        plt.plot(self.rewards_log, color='tab:blue', alpha=0.3, label='Raw Reward')  # 原始奖励，透明显示
        if len(self.rewards_log) > smooth_window:
            smooth = np.convolve(self.rewards_log, np.ones(smooth_window)/smooth_window, mode='same')
            valid_len = len(self.rewards_log) - smooth_window // 2
            smooth = smooth[:valid_len]
            plt.plot(range(valid_len), smooth, color='tab:orange', label=f'Smoothed ({smooth_window})')
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        if save_path:
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
            # filename = f"reward_curve_{timestamp}.png"
            filename = "reward_curve.png"
            plt.savefig(os.path.join(save_path, filename), dpi=300)
        plt.show()

        # --- Loss 曲线 ---
        plt.figure(figsize=(12, 5))
        plt.title("Total Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss Value")
        plt.plot(range(len(self.loss_log)), self.loss_log, label="Loss Total", color='tab:red')
        # if len(self.value_loss_log) > loss_smooth_window:
        #     smooth = np.convolve(self.value_loss_log, np.ones(loss_smooth_window)/loss_smooth_window, mode='same')
        #     plt.plot(range(len(self.value_loss_log)), smooth, color='tab:orange', label=f'Smoothed ({loss_smooth_window})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.8)
        if save_path:
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
            # filename = f"total_loss_curve_{timestamp}.png"
            filename = "total_loss_curve.png"
            plt.savefig(os.path.join(save_path, filename), dpi=300)
        plt.show()


# ===== 全局单例（保持原名与函数）=====
qlearning = MAPPOQoAR()


# --------- 实用工具：弹性解析 ---------
def _is_num(x):
    return isinstance(x, numbers.Number) and not isinstance(x, bool)

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


# --------- 模块级函数（保持旧名），增加弹性 ----------
def update_q_value(*args, **kwargs):
    sf, df, bf, current, next_hop, dest, band,reward = _normalize_update_args(*args, **kwargs)
    # print((f"[QoAR] set params failed: {sf, df, bf, current, next_hop, dest, band,reward}"))
    return qlearning.update_q_value(sf, df, bf, current, next_hop, dest, band,reward)


def update_lq(sf=None, ef=None, bf=None, current_node=None, next_hop=None, band=0, **kwargs):
    if sf is None or ef is None or bf is None:
        sf = 0.0 if sf is None else float(sf)
        ef = 0.0 if ef is None else float(ef)
        bf = 0.0 if bf is None else float(bf)
    if current_node is None:
        current_node = kwargs.get("current") or ""
    if next_hop is None:
        next_hop = kwargs.get("nh") or qlearning.get_best_next_hop(str(current_node)) or ""
    return qlearning.update_lq(float(sf), float(ef), float(bf), str(current_node), str(next_hop),band)


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

def plot_training_curves(save_path=None):
    qlearning.plot_training_curves(save_path=save_path)

# ===== 参数归一化/应用（静默）=====
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

def _apply_params(alpha, gamma, a, b, c):
    try:
        qlearning.set_parameters(alpha, gamma, a, b, c)
        for pg in qlearning.opt_pi.param_groups:
            pg["lr"] = qlearning.pi_lr
        for pg in qlearning.opt_v.param_groups:
            pg["lr"] = qlearning.vf_lr
        # print(f"[QoAR] params: alpha={alpha}, gamma={gamma}, a={a}, b={b}, c={c}")
        return True
    except Exception as e:
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

def random_action() -> int:

    return random.randint(0, 1)

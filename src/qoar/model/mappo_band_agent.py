"""
MAPPO-based per-hop band selector (inference only).

Observation (len >= 8):
  [0] band_now (0=2.4G, 1=5G)
  [1] thr24 (Mbps)
  [2] thr5  (Mbps)
  [3] pdr24 (0~1)
  [4] pdr5  (0~1)
  [5] dly24 (seconds)
  [6] dly5  (seconds)
  [7] speed (m/s)
Optional (len >= 10):
  [8] deg24 (neighbor consistency / count or ratio)
  [9] deg5

Returns list[int]: action per obs (0=2.4G, 1=5G)

Usage (from C++ via Python/C API):
  >>> init_agent(8)
  >>> choose_actions([[0, 1.2, 5.4, 0.9, 0.8, 0.05, 0.07, 12.3]])
"""
from __future__ import annotations

import math
import os
from typing import List, Sequence

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Heuristic gates and weights (tune if needed)1
# -----------------------------------------------------------------------------
THR_MIN_24: float = 0.05   # Mbps, minimal throughput considered viable on 2.4G
THR_MIN_5: float  = 0.10   # Mbps, minimal throughput considered viable on 5G
PDR_MIN: float     = 0.30  # minimal packet delivery ratio
DLY_MAX: float     = 0.20  # seconds, maximal acceptable delay

HYSTERESIS: float  = 0.15  # score margin required to flip the band

W_THR: float = 1.00  # throughput weight
W_PDR: float = 0.50  # delivery ratio weight
W_DLY: float = 0.30  # delay penalty weight
W_DEG: float = 0.25  # neighbor-consistency bias weight (optional features)


def _score(thr: float, pdr: float, dly: float, deg_bonus: float = 0.0) -> float:
    """Compute scalar preference score for one band.

    Uses log1p(thr) to compress very large throughputs, adds PDR, subtracts delay,
    and adds an optional neighbor-consistency bonus.


    """
    return (
        W_THR * math.log1p(max(thr, 0.0))
        + W_PDR * max(pdr, 0.0)
        - W_DLY * max(dly, 0.0)
        + deg_bonus
    )


def _viable(is5: bool, thr: float, pdr: float, dly: float) -> bool:
    """Hard gate for band viability.

    Passes if basic QoS is met, or if PDR is much better and delay much lower.
    """
    thr_min = THR_MIN_5 if is5 else THR_MIN_24
    if thr >= thr_min and pdr >= PDR_MIN and dly <= DLY_MAX:
        return True
    if pdr >= (PDR_MIN + 0.2) and dly <= (DLY_MAX * 0.7):
        return True
    return False


# -----------------------------------------------------------------------------
# Tiny policy network (only used for inference-time nudging)
# -----------------------------------------------------------------------------
class Policy(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128, n_actions: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # raw logits


class MappoBandAgent:
    """Inference-only agent.

    Loads optional weights from MAPPO_BAND_WEIGHTS or "mappo_band.pt".
    If weights are missing, falls back to heuristic with small NN nudging.
    """

    def __init__(self, obs_dim: int, n_actions: int = 2) -> None:
        self.policy = Policy(obs_dim, 128, n_actions)
        weight_path = os.environ.get("MAPPO_BAND_WEIGHTS", "mappo_band.pt")
        if os.path.exists(weight_path):
            self.policy.load_state_dict(torch.load(weight_path, map_location="cpu"))
            print('存在')
        self.policy.eval()

    @torch.no_grad()
    def choose_actions(self, obs_batch: Sequence[Sequence[float]]) -> List[int]:
        """Return 0=2.4G or 1=5G for each observation in obs_batch.

        Robust to obs length >= 8; if >=10, uses deg24/deg5 as consistency bias.
        
        """
        # Convert to tensor for NN pass; also keep Python list for rule parts.
        x = torch.tensor(obs_batch, dtype=torch.float32)
        logits = self.policy(x)

        actions: List[int] = []
        for i, o in enumerate(obs_batch):
            # Required features 速度呢？？
            band_now = int(round(float(o[0])))  # 0 or 1
            thr24 = float(o[1])
            thr5  = float(o[2])
            pdr24 = float(o[3])
            pdr5  = float(o[4])
            d24   = float(o[5])
            d5    = float(o[6])

            # Optional neighbor-consistency
            deg24 = float(o[8]) if len(o) >= 9 else 0.0
            deg5v = float(o[9]) if len(o) >= 10 else 0.0
            deg_sum = max(deg24 + deg5v, 1e-6)
            deg_bias_24 = W_DEG * ((deg24 - deg5v) / deg_sum)
            deg_bias_5  = -deg_bias_24

            # Hard gates
            v24 = _viable(False, thr24, pdr24, d24)
            v5  = _viable(True,  thr5,  pdr5,  d5)

            # Heuristic scores
            s24_rule = _score(thr24, pdr24, d24, deg_bias_24)
            s5_rule  = _score(thr5,  pdr5,  d5, deg_bias_5)

            # Tiny NN nudging (stable even without weights) 只是微调？？
            probs = torch.softmax(logits[i], dim=-1).cpu().tolist()
            s24 = s24_rule + 0.05 * (probs[0] - 0.5)
            s5  = s5_rule  + 0.05 * (probs[1] - 0.5)

            # Decision with viability and hysteresis
            if v24 and not v5:
                actions.append(0)
                continue
            if v5 and not v24:
                actions.append(1)
                continue
            if not v24 and not v5:
                actions.append(band_now)
                continue

            if band_now == 0:
                actions.append(1 if (s5 - s24) >= HYSTERESIS else 0)
            else:
                actions.append(0 if (s24 - s5) >= HYSTERESIS else 1)

            # Very close tie-breakers
            if abs(s5 - s24) < 0.05:
                if thr5 > thr24 * 1.2 and d5 <= d24 * 0.9 and v5:
                    actions[-1] = 1
                elif thr24 > thr5 * 1.2 and d24 < d5 * 0.9 and v24:
                    actions[-1] = 0

        return actions


# -----------------------------------------------------------------------------
# Module-level helpers for C++ bridge
# -----------------------------------------------------------------------------
_agent: MappoBandAgent | None = None


def init_agent(obs_dim: int, n_actions: int = 2) -> bool:
    """Initialize singleton agent (idempotent)."""
    global _agent
    if _agent is None:
        _agent = MappoBandAgent(obs_dim, n_actions)
    return True


def choose_actions(obs_batch: Sequence[Sequence[float]]) -> List[int]:
    """C++ bridge: choose band per observation."""
    global _agent
    assert _agent is not None, "Call init_agent first"
    return _agent.choose_actions(obs_batch)


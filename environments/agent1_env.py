# ─────────────────────────────────────────────
# environments/agent1_env.py  —  Stage 5 (Agent 1)
#
# Risk-averse agent.  Assets: 12 equities + TLT.
#
# Reward = log_return
#          − λ_drawdown   * drawdown_from_peak
#          − λ_volatility * rolling_portfolio_volatility
#          − turnover_cost * turnover
#
# The drawdown and volatility penalties push the agent toward:
#   • holding TLT during equity drawdowns (safe haven rotation)
#   • maintaining diversified weights (vol penalty punishes concentration)
#   • cutting losing positions quickly (drawdown penalty makes holding them costly)
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd

from config import LAMBDA_DRAWDOWN, LAMBDA_VOLATILITY, TURNOVER_COST
from environments.base_env import BaseTradingEnv


class Agent1Env(BaseTradingEnv):

    def __init__(
        self,
        features:         pd.DataFrame,
        prices:           pd.DataFrame,
        lambda_drawdown:  float = LAMBDA_DRAWDOWN,
        lambda_volatility: float = LAMBDA_VOLATILITY,
        turnover_cost:    float = TURNOVER_COST,
        **kwargs,
    ):
        super().__init__(features, prices, **kwargs)

        self.lambda_drawdown   = lambda_drawdown
        self.lambda_volatility = lambda_volatility
        self.turnover_cost     = turnover_cost

        # Keeps the last 20 daily log returns to compute rolling portfolio volatility
        self._return_history: list[float] = []

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._return_history = []
        return obs, info

    def compute_reward(
        self,
        log_return:  float,
        turnover:    float,
        new_weights: np.ndarray,
    ) -> float:
        """
        r = log_return
            − λ_drawdown   * D_t               (D_t = fraction below peak)
            − λ_volatility * σ_portfolio        (rolling 20-day std of log returns)
            − turnover_cost * turnover

        Tuning guide:
          λ_drawdown   too high → agent parks everything in TLT, near-zero return
          λ_volatility too high → agent holds equal weights and never concentrates
          Aim: Agent 1 Sharpe ≈ Agent 2 Sharpe, but Agent 1 max-drawdown ≈ half
        """
        # ── Drawdown penalty ─────────────────────────────────────────────────
        # Continuous penalty: always active as long as we are below the peak.
        # Resets per episode, not globally — keeps gradients meaningful early in training.
        drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-8)

        # ── Portfolio volatility penalty ─────────────────────────────────────
        # Rolling std of the agent's own realised log returns (last 20 days).
        self._return_history.append(log_return)
        if len(self._return_history) > 20:
            self._return_history.pop(0)

        portfolio_vol = float(np.std(self._return_history)) if len(self._return_history) > 1 else 0.0

        # ── Combine ──────────────────────────────────────────────────────────
        reward = (
            log_return
            - self.lambda_drawdown   * drawdown
            - self.lambda_volatility * portfolio_vol
            - self.turnover_cost     * turnover
        )

        return float(reward)

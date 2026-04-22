# ─────────────────────────────────────────────
# test_stages.py  —  smoke test for stages 1–5
# Run: python test_stages.py
# ─────────────────────────────────────────────

from config import AGENT1_TICKERS, AGENT2_TICKERS
from data.pipeline import build_features, split_data
from environments.agent1_env import Agent1Env
from environments.agent2_env import Agent2Env

# ── Stages 1–3: data pipeline ────────────────
print("=== Stages 1–3: Data Pipeline ===")
feat1, prices1 = build_features(AGENT1_TICKERS)
feat2, prices2 = build_features(AGENT2_TICKERS)
splits1 = split_data(feat1, prices1)
splits2 = split_data(feat2, prices2)

# ── Stage 4–5: environments ───────────────────
print("\n=== Stages 4–5: Environments ===")

env1 = Agent1Env(splits1["train_features"], splits1["train_prices"])
env2 = Agent2Env(splits2["train_features"], splits2["train_prices"])

# Quick rollout: random actions for one episode
for env, name in [(env1, "Agent1"), (env2, "Agent2")]:
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break

    print(f"{name}: obs_shape={obs.shape}  steps={steps}  "
          f"total_reward={total_reward:.4f}  "
          f"final_portfolio_value={info['portfolio_value']:.4f}")

print("\nAll stages passed.")

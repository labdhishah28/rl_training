import pandas as pd
import numpy as np
import os
from loader import DataLoader
from execution_env import ExecutionEnv
from agent import ExecutionAgent
from baselines import BaselineModels
# from stable_baselines3.common.env_checker import check_env

def generate_regime_data(num_steps: int = 50000, seed: int = 42) -> pd.DataFrame:
    """
    Generates multi-regime LOB data to improve agent generalization.

    Simulates four distinct market regimes sequentially, each occupying
    approximately 25% of the total steps:

    1. Trending Up    — sustained positive drift (momentum / bull market)
    2. Trending Down  — sustained negative drift (sell-off / liquidation)
    3. Mean-Reverting — Ornstein-Uhlenbeck process (range-bound / consolidation)
    4. High Volatility— zero drift with large shocks (news event / VIX spike)

    Training on all four regimes prevents the agent from over-fitting to a
    single price dynamic and improves out-of-sample performance.

    Args:
        num_steps: Total number of tick-level data points to generate.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame: LOB-compatible DataFrame with columns
            ['Timestamp', 'Bid_Price', 'Ask_Price', 'Bid_Volume', 'Ask_Volume'].
    """
    rng = np.random.default_rng(seed)
    seg = num_steps // 4  # Size of each regime segment

    # ── Regime 1: Trending Up ────────────────────────────────────────────────
    # Positive drift with moderate noise (simulates a bull run)
    prices_up = np.cumsum(rng.normal(0.02, 0.05, seg)) + 100.0

    # ── Regime 2: Trending Down ──────────────────────────────────────────────
    # Negative drift, picks up where regime 1 ended
    prices_down = np.cumsum(rng.normal(-0.02, 0.05, seg)) + prices_up[-1]

    # ── Regime 3: Mean-Reverting (Ornstein-Uhlenbeck) ───────────────────────
    # Price is pulled back toward a long-term mean (mu) over time
    theta = 0.10          # Mean-reversion speed
    mu    = prices_down[-1]  # Target mean (start of this segment's price)
    sigma_ou = 0.03       # OU noise amplitude
    prices_ou = np.empty(seg)
    prices_ou[0] = prices_down[-1]
    for i in range(1, seg):
        prices_ou[i] = (prices_ou[i - 1]
                        + theta * (mu - prices_ou[i - 1])
                        + sigma_ou * rng.normal())

    # ── Regime 4: High Volatility / Shock ───────────────────────────────────
    # Zero drift, large random shocks (simulates a news-driven market)
    prices_hv = np.cumsum(rng.normal(0.0, 0.15, seg)) + prices_ou[-1]

    # Combine all regimes into a single price series
    mid_prices = np.concatenate([prices_up, prices_down, prices_ou, prices_hv])

    # Spreads — tighter in trending regimes, much wider in high-vol regime
    spreads = np.abs(rng.normal(0.05, 0.01, num_steps))
    spreads[seg * 3:] = np.abs(rng.normal(0.15, 0.03, seg))  # HV = wider spread

    bid_prices = mid_prices - spreads / 2.0
    ask_prices = mid_prices + spreads / 2.0

    bid_volumes = rng.integers(100, 1000, num_steps)
    ask_volumes = rng.integers(100, 1000, num_steps)

    df = pd.DataFrame({
        'Timestamp': pd.date_range(start='2026-01-01', periods=num_steps, freq='s'),
        'Bid_Price': bid_prices,
        'Ask_Price': ask_prices,
        'Bid_Volume': bid_volumes,
        'Ask_Volume': ask_volumes,
    })
    return df


# ── Backward-compatible alias ────────────────────────────────────────────────
def generate_dummy_data(num_steps: int = 5000) -> pd.DataFrame:
    """Legacy single-regime random walk. Use generate_regime_data() for training."""
    return generate_regime_data(num_steps=num_steps)

def main():
    print("========================================")
    print(" Phase A: Multi-Regime Data Setup       ")
    print("========================================")
    data_path = "regime_market_data.csv"

    # Generate or load the multi-regime dataset
    if not os.path.exists(data_path):
        print("Generating multi-regime LOB data (50,000 steps across 4 regimes)...")
        df_raw = generate_regime_data(num_steps=50000)
        df_raw.to_csv(data_path, index=False)
        print(f"Saved to {data_path}.")

    loader = DataLoader(data_path)
    df = loader.load_data()
    print(f"Data loaded. Length: {len(df)} rows — "
          f"Trending Up / Down / Mean-Rev / High-Vol, ~{len(df)//4} rows each.")

    # ── Environment Parameters ────────────────────────────────────────────────
    # Liquidate 10,000 units over 100 steps (equities) or 0.1 BTC equivalent
    time_horizon      = 100
    initial_inventory = 10000.0

    # ── Evaluation Environment (single, non-vectorised) ───────────────────────
    eval_env_data = ExecutionEnv(
        data=df,
        initial_inventory=initial_inventory,
        time_horizon=time_horizon,
        kappa=0.01,
        gamma=0.005,
        is_buy=False
    )

    print("\n========================================")
    print(" Phase A: RL Agent Training (MLP) ")
    print("========================================")

    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback, CallbackList
    )

    def make_env():
        return ExecutionEnv(
            data=df,
            initial_inventory=initial_inventory,
            time_horizon=time_horizon,
            kappa=0.01,
            gamma=0.005,
            is_buy=False
        )

    # ── MLP Agent — vectorised training ──────────────────────────────────────
    vec_env_mlp = make_vec_env(make_env, n_envs=8)
    eval_env    = make_env()

    mlp_ckpt_cb = CheckpointCallback(
        save_freq=20000, save_path='./logs/mlp_checkpoints/', name_prefix='mlp_model'
    )
    mlp_eval_cb = EvalCallback(
        eval_env,
        best_model_save_path='./logs/mlp_best_model/',
        log_path='./logs/mlp_eval/',
        eval_freq=20000,
        deterministic=True,
        render=False
    )
    callback_mlp = CallbackList([mlp_ckpt_cb, mlp_eval_cb])

    print("\n--- Training MLP Agent (300,000 timesteps) ---")
    agent_mlp = ExecutionAgent(
        env=vec_env_mlp,
        model_path="ppo_execution_model_mlp",
        tensorboard_log="./logs/mlp_tb/",
        learning_rate=0.002338022245774756,
        ent_coef=0.005,          # Reduced: less random exploration → more deliberate pacing
        batch_size=64,
        gamma=0.9856900323768848,
        n_steps=2048             # Longer rollouts: agent sees full-episode consequences
    )
    agent_mlp.train(total_timesteps=300000, callback=callback_mlp)
    agent_mlp.save()


    

    
    print("\n========================================")
    print(" Phase A: RL Agent Evaluation vs Baselines ")
    print("========================================")

    def evaluate_agent(agent, env_to_eval, name):
        obs, _info = env_to_eval.reset()
        done = False
        total_reward = 0
        trajectory = []

        while not done:
            action, _ = agent.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env_to_eval.step(action)
            total_reward += reward
            trajectory.append(info['inventory_remaining'])
            done = terminated or truncated

        cash_flow = total_reward * 10000.0
        print(f"  {name:30s} Total Cash Flow: ${cash_flow:>15,.2f}")
        return trajectory, cash_flow

    # Fresh evaluation environments
    eval_env_mlp  = ExecutionEnv(data=df, initial_inventory=initial_inventory,
                                 time_horizon=time_horizon, kappa=0.01, gamma=0.005)

    print("\nEvaluating agents...")
    trajectory_mlp, cash_flow_mlp  = evaluate_agent(agent_mlp,  eval_env_mlp,  "MLP Agent")

    # Analytical baselines
    twap_traj = BaselineModels.get_twap_trajectory(initial_inventory, time_horizon)
    ac_traj   = BaselineModels.get_almgren_chriss_trajectory(
        initial_inventory, time_horizon, risk_aversion=1e-4, kappa=0.01
    )

    print(f"\nInventory remaining after 10 steps:")
    print(f"  TWAP Baseline:            {twap_traj[9]:>10.2f} units")
    print(f"  Almgren-Chriss (Optimal): {ac_traj[9]:>10.2f} units")
    print(f"  MLP RL Agent:             {trajectory_mlp[9]:>10.2f} units")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 7))
    plt.plot(twap_traj,       label='TWAP Baseline',       linestyle='--',  color='gray')
    plt.plot(ac_traj,         label='Almgren-Chriss (AC)', linestyle='-.',  color='orange')
    plt.plot(trajectory_mlp,  label='RL Agent (MLP)',      linestyle='-',   color='steelblue')
    plt.title('Execution Trajectories — Multi-Regime Training (Phase A)', fontsize=14)
    plt.xlabel('Time Step')
    plt.ylabel('Remaining Inventory (units)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('trajectories.png', dpi=150)
    print("\nSaved trajectories plot to trajectories.png")

    report = (
        "========================================\n"
        " RL Agent Evaluation Results\n"
        "========================================\n"
        f"MLP Agent Total Cash Flow: ${cash_flow_mlp:,.2f}\n\n"
        "Inventory remaining after 10 steps:\n"
        f"  TWAP Baseline:            {twap_traj[9]:>10.2f} units\n"
        f"  Almgren-Chriss (Optimal): {ac_traj[9]:>10.2f} units\n"
        f"  MLP RL Agent:             {trajectory_mlp[9]:>10.2f} units\n"
    )
    with open("evaluation_results.txt", "w") as f:
        f.write(report)
    print("Saved numerical results to evaluation_results.txt")

if __name__ == "__main__":
    main()

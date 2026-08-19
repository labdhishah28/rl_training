"""
live_trade.py
──────────────
Phase C: Paper Trading Orchestrator

Connects the trained RL agent to the Binance live WebSocket feed and runs
a full trade execution episode in paper-trading mode (no real money).

Every step is logged to a time-stamped CSV file for post-trade analysis.

Usage
-----
    # Using the default MLP model on BTCUSDT
    python live_trade.py

    # Full options
    python live_trade.py --symbol ETHUSDT --model ppo_execution_model_mlp \
                         --inventory 0.05 --horizon 100 --warmup 30

Arguments
---------
    --symbol    : Binance pair to trade (default: BTCUSDT)
    --model     : Path to the saved .zip model (without extension)
    --inventory : Starting inventory in base units, e.g. 0.05 BTC (default: 0.05)
    --horizon   : Number of execution steps (default: 100)
    --warmup    : Minimum ticks to collect before starting (default: 30)
    --kappa     : Temporary market impact coefficient (default: 0.00001)
    --gamma     : Permanent market impact coefficient (default: 0.000005)
    --buy       : Flag — buy (acquire) instead of sell (liquidate)
"""

import argparse
import csv
import time
import os

import numpy as np
import pandas as pd

from streaming_feed import BinanceStreamFeed, RollingBuffer
from execution_env import ExecutionEnv

try:
    from stable_baselines3 import PPO
except ImportError as e:
    raise ImportError(f"Missing dependency: {e}\nRun: pip install stable-baselines3")


# ──────────────────────────────────────────────────────────────────────────────
# Paper Trading Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_paper_trade(
    symbol: str       = "BTCUSDT",
    model_path: str   = "ppo_execution_model_mlp",
    initial_inventory: float = 0.05,   # in base asset (e.g. 0.05 BTC)
    time_horizon: int = 100,
    warmup_ticks: int = 30,
    kappa: float      = 0.00001,       # Impact params are tiny for crypto
    gamma: float      = 0.000005,
    is_buy: bool      = False,         # False = sell/liquidate, True = buy/acquire
) -> None:
    """
    Run one full paper-trading episode using the trained RL agent.

    Flow
    ----
    1. Start the Binance WebSocket feed → fills RollingBuffer
    2. Wait for ``warmup_ticks`` ticks before touching the env
    3. Snapshot the buffer to build a seed DataFrame (used to size obs/action spaces)
    4. Create ExecutionEnv in live mode (live_feed=buffer)
    5. Load the trained PPO/LSTM model
    6. Step loop: wait for tick → predict → step → log → repeat
    7. Stop feed, save CSV log

    Args:
        symbol            : Binance trading pair.
        model_path        : Path to the SB3 .zip model (without extension).
        initial_inventory : Starting inventory in base units.
        time_horizon      : Maximum number of execution steps.
        warmup_ticks      : Number of ticks to collect before starting.
        kappa             : Temporary impact coefficient.
        gamma             : Permanent impact coefficient.
        is_buy            : If True, agent acquires inventory; if False, liquidates.
    """
    mode_str = "BUY (acquire)" if is_buy else "SELL (liquidate)"
    print("=" * 60)
    print(f"  PAPER TRADING — {symbol}")
    print(f"  Model     : {model_path}")
    print(f"  Inventory : {initial_inventory} (base units)")
    print(f"  Horizon   : {time_horizon} steps")
    print(f"  Mode      : {mode_str}")
    print(f"  kappa     : {kappa}  |  gamma: {gamma}")
    print("=" * 60)

    # ── 1. Start live feed ────────────────────────────────────────────────────
    buffer = RollingBuffer(maxlen=500)
    feed   = BinanceStreamFeed(symbol=symbol, buffer=buffer)
    feed.start()

    # ── 2. Buffer warm-up ─────────────────────────────────────────────────────
    print(f"\n[*] Waiting for {warmup_ticks} ticks to warm up buffer...")
    timeout_start = time.time()
    while len(buffer) < warmup_ticks:
        if time.time() - timeout_start > 60:
            feed.stop()
            raise TimeoutError(
                f"Could not collect {warmup_ticks} ticks within 60 s. "
                "Check your internet connection or try a different symbol."
            )
        time.sleep(0.1)
    print(f"[*] Buffer warmed up ({len(buffer)} ticks). Starting execution...\n")

    # ── 3. Seed DataFrame (for obs/action space sizing) ───────────────────────
    seed_df = buffer.as_dataframe()
    # Ensure the DataFrame has at least time_horizon rows so the env can reset
    if len(seed_df) < time_horizon + 2:
        seed_df = pd.concat([seed_df] * ((time_horizon // len(seed_df)) + 2),
                             ignore_index=True)

    # ── 4. Create live ExecutionEnv ───────────────────────────────────────────
    env = ExecutionEnv(
        data=seed_df,
        initial_inventory=initial_inventory,
        time_horizon=time_horizon,
        kappa=kappa,
        gamma=gamma,
        is_buy=is_buy,
        live_feed=buffer,          # ← Live mode enabled
    )

    # ── 5. Load model ─────────────────────────────────────────────────────────
    model_file = f"{model_path}.zip"
    if not os.path.exists(model_file):
        feed.stop()
        raise FileNotFoundError(
            f"Model file '{model_file}' not found. "
            "Train the agent first with: python train.py"
        )

    model = PPO.load(model_path, env=env)

    print(f"[*] Loaded model from {model_file}")

    # ── 6. Step loop ──────────────────────────────────────────────────────────
    obs, _ = env.reset()
    done          = False
    log_rows      = []
    step_num      = 0

    print(f"\n{'Step':>5}  {'Mid Price':>12}  {'Action':>8}  {'Vol Traded':>12}  {'Inventory':>12}  {'OBI':>8}")
    print("-" * 70)

    while not done:
        # Wait for a fresh tick (max 5 s before giving up)
        arrived = buffer.wait_for_tick(timeout=5.0)
        if not arrived:
            print("[!] Tick timeout — no data from Binance for 5 s. Stopping.")
            break

        # Predict action
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Capture current tick for logging
        tick = buffer.latest()
        mid  = tick["Mid_Price"] if tick else float("nan")
        obi  = tick["OBI"]       if tick else float("nan")

        step_num += 1
        print(
            f"{step_num:>5}  {mid:>12.4f}  {float(action[0]):>8.4f}  "
            f"{info['volume_traded']:>12.6f}  {info['inventory_remaining']:>12.6f}  "
            f"{obi:>8.4f}"
        )

        log_rows.append({
            "step"               : step_num,
            "timestamp"          : tick["timestamp"] if tick else time.time(),
            "mid_price"          : mid,
            "obi"                : obi,
            "action"             : float(action[0]),
            "volume_traded"      : info["volume_traded"],
            "inventory_remaining": info["inventory_remaining"],
            "execution_price"    : info["execution_price"],
            "temp_impact"        : info["temporary_impact"],
            "perm_impact_acc"    : info["permanent_impact_acc"],
            "reward"             : reward,
        })

        done = terminated or truncated

    # ── 7. Stop feed and save log ─────────────────────────────────────────────
    feed.stop()

    # Guard: nothing executed (e.g. WebSocket immediately timed out)
    if not log_rows:
        print("\n[!] No steps were executed — nothing to save.")
        return

    total_cash = sum(r["reward"] for r in log_rows) * 10000.0
    print("\n" + "=" * 60)
    print(f"  Trade complete after {step_num} steps")
    print(f"  Inventory remaining : {log_rows[-1]['inventory_remaining']:.6f}")
    print(f"  Estimated cash flow : ${total_cash:,.4f}")
    print("=" * 60)

    if log_rows:
        log_file = f"live_trade_{symbol}_{int(time.time())}.csv"
        with open(log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"\n[*] Trade log saved to {log_file}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

import pandas as pd  # noqa: E402  (deferred to avoid circular at top)

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RL Paper Trading via Binance WebSocket",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol",    default="BTCUSDT",                 help="Binance pair")
    p.add_argument("--model",     default="ppo_execution_model_mlp", help="Model path (no .zip)")
    p.add_argument("--inventory", type=float, default=0.05,          help="Starting inventory (base units)")
    p.add_argument("--horizon",   type=int,   default=100,           help="Execution time horizon (steps)")
    p.add_argument("--warmup",    type=int,   default=30,            help="Buffer warm-up ticks before start")
    p.add_argument("--kappa",     type=float, default=0.00001,       help="Temporary market impact coefficient")
    p.add_argument("--gamma",     type=float, default=0.000005,      help="Permanent market impact coefficient")
    p.add_argument("--buy",       action="store_true",               help="Buy/acquire mode instead of sell/liquidate")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_paper_trade(
        symbol=args.symbol,
        model_path=args.model,
        initial_inventory=args.inventory,
        time_horizon=args.horizon,
        warmup_ticks=args.warmup,
        kappa=args.kappa,
        gamma=args.gamma,
        is_buy=args.buy,
    )

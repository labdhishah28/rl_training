# Optimal Execution Engine: Phase 6 Documentation

## Live Streaming & Paper Trading

**Phase 6** connects the trained RL agent to real-time Binance WebSocket data
and runs full paper-trading episodes with no synthetic data involved.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Main Thread                             │
│                                                                 │
│   live_trade.py                                                 │
│   ┌──────────────┐    obs     ┌──────────────────────────┐      │
│   │ ExecutionEnv │ ◄───────── │   ExecutionAgent (PPO)   │      │
│   │ (live mode)  │ ──────────►│   model.predict(obs)     │      │
│   └──────┬───────┘   action   └──────────────────────────┘      │
│          │ reads                                                 │
│          ▼                                                       │
│   ┌──────────────┐                                              │
│   │ RollingBuffer│ ◄── thread-safe deque (maxlen=500)           │
│   └──────┬───────┘                                              │
│          │ push                                                  │
└──────────┼──────────────────────────────────────────────────────┘
           │
┌──────────┼──────────────────────────────────────────────────────┐
│  Background Daemon Thread (asyncio event loop)                  │
│          │                                                       │
│   ┌──────▼───────────────────────────────────────────────────┐  │
│   │  BinanceStreamFeed                                       │  │
│   │  wss://stream.binance.com:9443/ws/btcusdt@bookTicker     │  │
│   │  Auto-reconnects on disconnect                           │  │
│   └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

The two threads never block each other:
- The **feed thread** (`asyncio` + `websockets`) pushes ticks into the buffer
- The **main thread** reads the latest tick on every `env.step()` call

---

## RollingBuffer

A thread-safe, bounded circular buffer of tick dictionaries.

```python
from streaming_feed import RollingBuffer

buf = RollingBuffer(maxlen=500)
buf.push({"Mid_Price": 65000.0, "Spread": 5.0, "OBI": 0.12, ...})

tick = buf.latest()          # most recent tick
df   = buf.as_dataframe()    # snapshot as DataFrame
arrived = buf.wait_for_tick(timeout=5.0)  # block until next push
```

### Thread Safety

All read/write operations are protected by a `threading.Lock()`.
A `threading.Event` is set-then-cleared on every push so `wait_for_tick()`
can efficiently block without polling.

---

## BinanceStreamFeed

Connects to the Binance **Best Bid/Ask Ticker** stream:

```
wss://stream.binance.com:9443/ws/{symbol}@bookTicker
```

Each message delivers the best bid/ask price and quantity in real-time:

```json
{
  "b": "64850.00",   // best bid price
  "B": "0.451",      // best bid quantity
  "a": "64852.00",   // best ask price
  "A": "0.123"       // best ask quantity
}
```

The feed processes each message into the standard tick schema:

| Field | Derivation |
|---|---|
| `Mid_Price` | `(bid + ask) / 2` |
| `Spread` | `ask − bid` |
| `OBI` | `(bid_vol − ask_vol) / (bid_vol + ask_vol)` |
| `Bid_Volume` | Raw from WebSocket |
| `Ask_Volume` | Raw from WebSocket |

### Auto-Reconnect

If the WebSocket connection drops (network hiccup, Binance rolling restart), the feed
sleeps for `reconnect_delay` seconds and reconnects automatically.
The main thread continues to read stale buffer data during this window.

---

## ExecutionEnv — Live Mode

The `live_feed` parameter activates live mode.  In this mode:

| Method | Static Mode | Live Mode |
|---|---|---|
| `_get_obs()` | Reads `self.mid_prices[step]` | Reads `live_feed.latest()['Mid_Price']` |
| `step()` | Reads indexed array | Reads `live_feed.latest()` |
| Episode reset | Picks random start offset | Resets counters only |
| Buffer fallback | N/A | Falls back to last static array row if buffer is empty |

> [!IMPORTANT]
> **Never use `live_feed` during PPO training.**  The SB3 `learn()` method calls
> `env.reset()` frequently and expects consistent episodes.  Live mode is for
> **inference only** — the model must already be trained before running
> `live_trade.py`.

---

## live_trade.py — Paper Trading Orchestrator

### Usage

```bash
# Default: BTCUSDT, MLP model, 0.05 BTC, 100 steps
python live_trade.py

# Ethereum with LSTM model, 0.1 ETH inventory, 50 steps
python live_trade.py --symbol ETHUSDT --model ppo_execution_model_lstm \
                     --inventory 0.1 --horizon 50 --warmup 30 --lstm
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--symbol` | `BTCUSDT` | Binance trading pair |
| `--model` | `ppo_execution_model_mlp` | Saved model path (no `.zip`) |
| `--inventory` | `0.05` | Starting inventory in base units (e.g. BTC) |
| `--horizon` | `100` | Max execution steps |
| `--warmup` | `30` | Ticks to collect before starting |
| `--lstm` | (flag) | Load as RecurrentPPO instead of PPO |

### Output: Console

```
Step    Mid Price    Action   Vol Traded    Inventory     OBI
─────────────────────────────────────────────────────────────────
   1  64850.0000    0.0512    0.002560    0.047440    0.1234
   2  64855.5000    0.0489    0.002318    0.045122    0.0891
   ...
```

### Output: CSV Log

A CSV file named `live_trade_{SYMBOL}_{unix_timestamp}.csv` is written on completion:

| Column | Description |
|---|---|
| `step` | Step number |
| `timestamp` | Unix epoch (seconds) |
| `mid_price` | Mid price at time of decision |
| `obi` | Order book imbalance |
| `action` | Fraction of remaining inventory to trade (0–1) |
| `volume_traded` | Actual units traded this step |
| `inventory_remaining` | Units still to execute |
| `execution_price` | Price after spread + temporary impact |
| `temp_impact` | Estimated temporary price impact |
| `perm_impact_acc` | Accumulated permanent price impact |
| `reward` | Scaled reward received |

---

## End-to-End Workflow

```
Phase 4 (Dummy)          Phase 5 (Historical)        Phase 6 (Live)
────────────────         ──────────────────────       ─────────────────────
python train.py   →      train.py with Binance  →    python live_trade.py
                         DataLoader.from_binance()
Produces:                Produces:                    Produces:
  ppo_execution_         improved model weights       live_trade_*.csv
  model_mlp.zip                                       console output
  ppo_execution_
  model_lstm.zip
```

---

## Paper vs Live Trading

This system is designed for **paper trading** — all decisions are logged but
no real orders are placed.  To move to live trading, you would need:

1. A Binance account with API key + secret
2. An order execution layer (e.g. `python-binance` library)
3. Replacing the `env.step()` loop with an actual `client.create_order()` call
4. Position tracking and risk controls

> [!CAUTION]
> **Never run live trading with untested models on mainnet funds.**
> Paper trade for at least several sessions, review the CSV logs, and
> verify that execution quality consistently beats TWAP before considering
> any live deployment.

---

## Installation Requirements

```bash
pip install websockets requests pandas numpy stable-baselines3 sb3-contrib pyarrow
```

| Package | Purpose |
|---|---|
| `websockets` | Binance WebSocket client (async) |
| `requests` | Binance REST API (historical fetch) |
| `pyarrow` | Parquet read/write for data cache |
| `sb3-contrib` | RecurrentPPO (LSTM agent) |

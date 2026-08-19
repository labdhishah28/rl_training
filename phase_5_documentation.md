# Optimal Execution Engine: Phase 5 Documentation

## Real Historical Data via Binance REST API

**Phase 5** replaces synthetic dummy data with genuine market microstructure
data fetched from the **Binance spot REST API** — completely free, no API key
required.

---

## Why Real Historical Data?

| | Dummy Data (Phase 4) | Real Data (Phase 5) |
|---|---|---|
| Price process | Synthetic GBM / OU | Actual observed prices |
| OBI | Random volumes | Real taker buy/sell imbalance |
| Spread | Gaussian noise | Actual bid-ask dynamics |
| Liquidity patterns | Uniform | Intraday U-shaped curve, news spikes |
| Correlation structure | None | Cross-asset correlations preserved |

The agent trained on real data will encounter the market's true statistical
properties: fat tails, volatility clustering, and correlated OBI signals.

---

## Data Source: Binance Kline (Candlestick) API

**Endpoint**: `GET https://api.binance.com/api/v3/klines`

**Rate limits**: 1,200 requests/minute — we sleep 250 ms between pages.

**No authentication required** — this is a public endpoint.

### Kline Column Mapping

| Binance Column | Index | Description |
|---|---|---|
| `open_time` | 0 | Bar open timestamp (ms) → `Timestamp` |
| `high` | 2 | Candle high price |
| `low` | 3 | Candle low price |
| `volume` | 5 | Total base-asset volume traded |
| `taker_buy_base_asset_volume` | 9 | Volume from market buy orders |

### LOB Feature Derivation

```
Mid_Price  = (high + low) / 2
Spread     = (high − low) × 0.10     # 10 % of the intrabar price range
Bid_Price  = Mid_Price − Spread / 2
Ask_Price  = Mid_Price + Spread / 2
Bid_Volume = taker_buy_base_volume   # buyers aggressively crossing the ask
Ask_Volume = volume − Bid_Volume     # sellers aggressively crossing the bid
OBI        = (Bid_Vol − Ask_Vol) / (Bid_Vol + Ask_Vol)
```

> [!NOTE]
> **Why 10 % of the price range for spread?**
> For 1-minute BTCUSDT bars at a $65,000 price level, the typical bar range
> is $50–$200.  Scaling by 0.10 gives a synthetic spread of $5–$20 — consistent
> with realistic taker costs for large institutional orders.  For higher-frequency
> (1-second) bars, the scaling factor should be reduced to ~0.01.

---

## The OBI Signal on Real Data

The Binance `taker_buy_base_asset_volume` field represents orders that
**actively crossed the best ask** — i.e., aggressive buyers.  The complement
is aggressive sellers.

This gives us a real **Order Book Imbalance** signal:
- **OBI > 0**: Buy pressure dominates — favourable to sell (higher price)
- **OBI < 0**: Sell pressure dominates — favourable for buyers

Unlike the synthetic OBI (which is just random volume noise), the Binance OBI
has genuine predictive content and should meaningfully improve agent performance.

---

## Files Added / Changed

| File | Description |
|---|---|
| `real_data_loader.py` | New — `BinanceHistoricalLoader` class |
| `loader.py` | Updated — `DataLoader.from_binance()` factory classmethod |
| `btcusdt_1m_5000.parquet` | Auto-generated — 5,000 bar cache (or similar) |

---

## Usage Examples

### Quick Start — fetch and train

```python
from loader import DataLoader
from execution_env import ExecutionEnv

# Fetch 5,000 1-minute BTCUSDT bars (or load from cache)
loader = DataLoader.from_binance(
    symbol="BTCUSDT",
    interval="1m",
    n_bars=5000,
    cache_path="btcusdt_1m_5000.parquet"
)
df = loader.load_data()

env = ExecutionEnv(data=df, initial_inventory=0.05, time_horizon=100)
```

### Direct use of BinanceHistoricalLoader

```python
from real_data_loader import BinanceHistoricalLoader

loader = BinanceHistoricalLoader(symbol="ETHUSDT", interval="5m")
df = loader.load_or_fetch("ethusdt_5m.parquet", n_bars=2000)
print(df[['Timestamp', 'Mid_Price', 'Spread', 'OBI']].tail())
```

### Supported Symbols & Intervals

| Symbol | Description |
|---|---|
| `BTCUSDT` | Bitcoin / USDT (highest volume, tightest spreads) |
| `ETHUSDT` | Ethereum / USDT |
| `BNBUSDT` | Binance Coin / USDT |
| `SOLUSDT` | Solana / USDT |

| Interval | Use Case |
|---|---|
| `1m` | Standard training (recommended) |
| `5m` | Faster fetch, lower resolution |
| `1h` | Long-term regime analysis |
| `1s` | Ultra-HFT (very large data, use with caution) |

---

## Caching Strategy

`load_or_fetch()` implements a simple **read-through cache**:

```
if parquet file exists on disk:
    load and return immediately (no API call)
else:
    fetch from Binance REST → save to parquet → return
```

This means training runs are fully reproducible after the first fetch.
To refresh the data, simply delete the `.parquet` file.

---

## Transition from Dummy to Real Data

Your existing `ExecutionEnv` and `ExecutionAgent` code requires **zero changes**.
The real data loader produces the same schema as `DataLoader`:

```
Timestamp | Bid_Price | Ask_Price | Bid_Volume | Ask_Volume | Mid_Price | Spread | OBI
```

Simply swap the data source in `train.py`:

```python
# Before (Phase 4):
loader = DataLoader("regime_market_data.csv")
df = loader.load_data()

# After (Phase 5):
loader = DataLoader.from_binance(symbol="BTCUSDT", interval="1m",
                                  n_bars=5000, cache_path="btcusdt_1m.parquet")
df = loader.load_data()
```

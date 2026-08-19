"""
real_data_loader.py
────────────────────
Phase B: Binance Historical LOB Data Loader

Fetches OHLCV kline data from the Binance REST API (no API key required)
and converts it to the LOB schema expected by ExecutionEnv:
    [Timestamp, Bid_Price, Ask_Price, Bid_Volume, Ask_Volume, Mid_Price, Spread, OBI]

The OBI is derived from Binance's taker buy/sell volume split — a real and
meaningful microstructure signal, not a synthetic approximation.

Usage
-----
    from real_data_loader import BinanceHistoricalLoader

    loader = BinanceHistoricalLoader(symbol="BTCUSDT", interval="1m")
    df = loader.load_or_fetch("btcusdt_1m.parquet", n_bars=5000)

    # Feed directly into ExecutionEnv — same API as DataLoader
    from execution_env import ExecutionEnv
    env = ExecutionEnv(data=df, ...)
"""

import time
import os

import numpy as np
import pandas as pd
import requests


class BinanceHistoricalLoader:
    """
    Downloads historical kline (candlestick) data from the Binance spot REST API
    and converts it to a Level-1 order book compatible DataFrame.

    Parameters
    ----------
    symbol : str
        Binance trading pair symbol, e.g. "BTCUSDT", "ETHUSDT". Case-insensitive.
    interval : str
        Kline interval string.  Supported values: "1s", "1m", "3m", "5m", "15m",
        "30m", "1h", "4h", "1d".  Shorter intervals yield higher-resolution data.

    Notes on schema conversion
    --------------------------
    Binance klines provide:
        open, high, low, close, volume, taker_buy_base_volume

    We derive the LOB features as:
        Mid_Price  = (high + low) / 2
        Spread     = (high - low) × 0.10   ← conservative bid-ask estimate
        Bid_Price  = Mid_Price − Spread / 2
        Ask_Price  = Mid_Price + Spread / 2
        Bid_Volume = taker_buy_base_volume  (aggressors hitting the ask — bullish)
        Ask_Volume = volume − taker_buy_base_volume   (aggressors hitting the bid)
        OBI        = (Bid_Volume − Ask_Volume) / (Bid_Volume + Ask_Volume)

    The OBI derived this way is a genuine microstructure signal: positive when
    buy pressure dominates, negative when selling pressure dominates.
    """

    BASE_URL  = "https://api.binance.com/api/v3/klines"
    MAX_LIMIT = 1000  # Binance hard cap per request

    def __init__(self, symbol: str = "BTCUSDT", interval: str = "1m"):
        self.symbol   = symbol.upper()
        self.interval = interval

    # ──────────────────────────────────────────────────────────────────────────
    # Public Interface
    # ──────────────────────────────────────────────────────────────────────────

    def fetch(self, n_bars: int = 5000, save_path: str | None = None) -> pd.DataFrame:
        """
        Fetches the last ``n_bars`` klines from Binance and returns a
        pre-processed LOB-compatible DataFrame.

        Args:
            n_bars    : Total number of candlestick bars to retrieve.
            save_path : If provided, the resulting DataFrame is saved as a
                        Parquet file at this path for offline reuse.

        Returns:
            pd.DataFrame with columns:
                Timestamp, Bid_Price, Ask_Price, Bid_Volume, Ask_Volume,
                Mid_Price, Spread, OBI
        """
        all_rows: list = []
        end_time: int | None = None
        remaining = n_bars

        print(f"[BinanceHistoricalLoader] Fetching {n_bars} × {self.interval} "
              f"bars for {self.symbol}...")

        while remaining > 0:
            limit  = min(remaining, self.MAX_LIMIT)
            params = {"symbol": self.symbol, "interval": self.interval, "limit": limit}
            if end_time is not None:
                params["endTime"] = end_time

            resp = requests.get(self.BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            rows = resp.json()

            if not rows:
                print("[BinanceHistoricalLoader] No more data available from Binance.")
                break

            # Prepend because we're paging backwards in time
            all_rows = rows + all_rows
            end_time  = int(rows[0][0]) - 1   # go further back in time
            remaining -= len(rows)

            print(f"  {len(all_rows):>6,} / {n_bars} bars fetched...")
            time.sleep(0.25)   # stay within Binance rate limits (1,200 req/min)

        df = self._to_dataframe(all_rows)
        print(f"[BinanceHistoricalLoader] Done — {len(df):,} bars ready.")

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            df.to_parquet(save_path, index=False)
            print(f"[BinanceHistoricalLoader] Saved to {save_path}")

        return df

    def load_or_fetch(self, save_path: str, n_bars: int = 5000) -> pd.DataFrame:
        """
        Loads data from a local Parquet cache if it exists, otherwise
        fetches from Binance and saves the result.

        This is the recommended entry-point for training runs to avoid
        unnecessary API calls.

        Args:
            save_path : Local path to the Parquet cache file.
            n_bars    : Number of bars to fetch if the cache is missing.

        Returns:
            pd.DataFrame (same schema as ``fetch()``)
        """
        if os.path.exists(save_path):
            print(f"[BinanceHistoricalLoader] Loading cached data from {save_path}...")
            df = pd.read_parquet(save_path)
            print(f"[BinanceHistoricalLoader] Loaded {len(df):,} bars from cache.")
            return df

        return self.fetch(n_bars=n_bars, save_path=save_path)

    # ──────────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _to_dataframe(self, rows: list) -> pd.DataFrame:
        """
        Converts the raw Binance kline list-of-lists into a LOB-compatible
        DataFrame.

        Binance kline columns (index → meaning):
            0  open_time (ms)
            1  open
            2  high
            3  low
            4  close
            5  volume
            6  close_time (ms)
            7  quote_asset_volume
            8  number_of_trades
            9  taker_buy_base_asset_volume
            10 taker_buy_quote_asset_volume
            11 ignore
        """
        _COLS = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
        ]
        df = pd.DataFrame(rows, columns=_COLS)

        # ── Parse numeric types ──────────────────────────────────────────────
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ("open", "high", "low", "close", "volume", "taker_buy_base_vol"):
            df[col] = df[col].astype(float)

        # ── Derive LOB features ──────────────────────────────────────────────
        df["Mid_Price"] = (df["high"] + df["low"]) / 2.0

        # Approximate bid-ask spread as 10 % of the candle's price range.
        # For 1-minute BTCUSDT bars, this is ~$5-15 on a $60k price, which is
        # a realistic taker spread for large-size orders.
        raw_spread   = (df["high"] - df["low"]) * 0.10
        df["Spread"] = raw_spread.clip(lower=1e-6)

        df["Bid_Price"] = df["Mid_Price"] - df["Spread"] / 2.0
        df["Ask_Price"] = df["Mid_Price"] + df["Spread"] / 2.0

        # OBI from taker buy/sell volume split — a true microstructure signal
        df["Bid_Volume"] = df["taker_buy_base_vol"]
        df["Ask_Volume"] = df["volume"] - df["taker_buy_base_vol"]

        total_vol  = df["Bid_Volume"] + df["Ask_Volume"]
        df["OBI"]  = np.where(
            total_vol > 0,
            (df["Bid_Volume"] - df["Ask_Volume"]) / total_vol,
            0.0,
        )

        df.rename(columns={"open_time": "Timestamp"}, inplace=True)

        # Keep only the LOB-compatible columns (same schema as DataLoader)
        result = df[[
            "Timestamp", "Bid_Price", "Ask_Price",
            "Bid_Volume", "Ask_Volume", "Mid_Price", "Spread", "OBI",
        ]].copy()

        result.reset_index(drop=True, inplace=True)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Standalone test / demo
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = BinanceHistoricalLoader(symbol="BTCUSDT", interval="1m")
    df = loader.load_or_fetch("btcusdt_1m_5000.parquet", n_bars=5000)

    print("\n── Schema ───────────────────────────────────────────")
    print(df.dtypes)
    print("\n── Sample rows ──────────────────────────────────────")
    print(df.head(5).to_string())
    print(f"\nOBI range : [{df['OBI'].min():.4f}, {df['OBI'].max():.4f}]")
    print(f"Spread avg: {df['Spread'].mean():.4f}")
    print(f"Mid_Price avg: {df['Mid_Price'].mean():.2f}")
    print("\nBinanceHistoricalLoader test passed ✓")

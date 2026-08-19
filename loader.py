import pandas as pd
import numpy as np
import os

class DataLoader:
    """
    Data Pipeline / Parser for Optimal Trade Execution RL Environment.
    Parses historical tick or Level 2 order book data and calculates OBI.

    Supports two data sources:
        'file'    (default): Load from a local CSV or Parquet file.
        'binance'           : Fetch from the Binance REST API via
                              ``BinanceHistoricalLoader``.  Requires the
                              ``requests`` package.

    Example — load from file (existing behaviour):
        loader = DataLoader("dummy_market_data.csv")
        df = loader.load_data()

    Example — fetch from Binance:
        loader = DataLoader.from_binance(symbol="BTCUSDT", interval="1m",
                                         n_bars=5000,
                                         cache_path="btcusdt_1m.parquet")
        df = loader.load_data()   # returns already-processed DataFrame
    """
    def __init__(self, file_path: str, source: str = 'file'):
        """
        Args:
            file_path : Path to the local CSV/Parquet file  (used when source='file').
            source    : Data source — 'file' or 'binance'.
        """
        self.file_path = file_path
        self.source    = source
        self.data      = None

    # ── Factory: Binance convenience constructor ──────────────────────────────
    @classmethod
    def from_binance(
        cls,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        n_bars: int = 5000,
        cache_path: str = "binance_data.parquet",
    ) -> "DataLoader":
        """
        Create a DataLoader backed by real Binance historical data.

        The data is fetched (or loaded from the local Parquet cache) and
        pre-processed to the standard LOB schema before ``load_data()`` is
        called, so ``_preprocess()`` is intentionally skipped.

        Args:
            symbol     : Binance trading pair, e.g. "BTCUSDT".
            interval   : Kline interval string, e.g. "1m", "5m", "1h".
            n_bars     : Number of bars to fetch if the cache is missing.
            cache_path : Local path for the Parquet cache.

        Returns:
            DataLoader instance whose ``load_data()`` returns the real data.
        """
        from real_data_loader import BinanceHistoricalLoader
        binance_loader = BinanceHistoricalLoader(symbol=symbol, interval=interval)
        df = binance_loader.load_or_fetch(cache_path, n_bars=n_bars)

        # Wrap in a DataLoader instance; store the DataFrame directly so
        # load_data() can return it without re-reading from disk.
        instance = cls(file_path=cache_path, source='binance')
        instance.data = df
        return instance

    # ── Primary interface ─────────────────────────────────────────────────────

    def load_data(self) -> pd.DataFrame:
        """
        Loads and returns the pre-processed LOB DataFrame.

        For source='file', reads CSV/Parquet then calls ``_preprocess()``.
        For source='binance', returns the already-processed DataFrame that
        was set by ``from_binance()``.
        """
        if self.source == 'binance':
            if self.data is None:
                raise RuntimeError("Binance DataLoader has no data. Use DataLoader.from_binance().")
            return self.data

        # Default: load from local file
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found at {self.file_path}")

        if self.file_path.endswith('.csv'):
            self.data = pd.read_csv(self.file_path)
        elif self.file_path.endswith('.parquet'):
            self.data = pd.read_parquet(self.file_path)
        else:
            raise ValueError("Unsupported file format. Please use .csv or .parquet")

        self._preprocess()
        return self.data

    def _preprocess(self):
        """
        Calculates Mid-price, Spread, and Order Book Imbalance (OBI).
        Outputs a clean, time-indexed DataFrame.
        """
        if self.data is None or self.data.empty:
            raise ValueError("Data is empty or not loaded.")

        # Skip preprocessing if already done (e.g. Binance data has these cols)
        if all(c in self.data.columns for c in ('Mid_Price', 'Spread', 'OBI')):
            if 'Timestamp' in self.data.columns:
                self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
                self.data.sort_values('Timestamp', inplace=True)
            self.data.reset_index(drop=True, inplace=True)
            return
            
        # 1. Mid-price: (Bid + Ask) / 2
        self.data['Mid_Price'] = (self.data['Bid_Price'] + self.data['Ask_Price']) / 2.0
        
        # 2. Spread: Ask - Bid
        self.data['Spread'] = self.data['Ask_Price'] - self.data['Bid_Price']
        
        # 3. Order Book Imbalance (OBI) based on volumes at the top of the book
        # Formula: (Bid_Volume - Ask_Volume) / (Bid_Volume + Ask_Volume)
        total_volume = self.data['Bid_Volume'] + self.data['Ask_Volume']
        
        # Avoid division by zero by setting OBI to 0 where total_volume is 0
        self.data['OBI'] = np.where(
            total_volume > 0,
            (self.data['Bid_Volume'] - self.data['Ask_Volume']) / total_volume,
            0.0
        )
        
        # Ensure timestamp is datetime and sorted
        if 'Timestamp' in self.data.columns:
            self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
            self.data.sort_values('Timestamp', inplace=True)
            
        # Reset index to ensure it is continuous for the Gym environment
        self.data.reset_index(drop=True, inplace=True)

if __name__ == "__main__":
    # Example usage for testing
    print("DataLoader module loaded. Use it by importing DataLoader.")


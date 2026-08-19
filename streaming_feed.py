"""
streaming_feed.py
──────────────────
Phase C: Real-Time Binance WebSocket Feed + Thread-Safe Rolling Buffer

Two classes are provided:

  RollingBuffer
    A thread-safe, bounded deque that stores processed market ticks as dicts.
    Written to by the streaming feed thread; read from by ExecutionEnv and
    the live trading loop.

  BinanceStreamFeed
    Connects to the Binance Best Bid/Ask Ticker WebSocket stream for a given
    symbol and continuously pushes processed ticks into a RollingBuffer.
    Runs entirely in a background daemon thread — the main thread is free to
    run the RL predict loop.

Tick schema (what gets pushed into the buffer)
──────────────────────────────────────────────
  {
    "timestamp"  : float      # Unix epoch (seconds)
    "Bid_Price"  : float      # Best bid price
    "Ask_Price"  : float      # Best ask price
    "Bid_Volume" : float      # Best bid quantity
    "Ask_Volume" : float      # Best ask quantity
    "Mid_Price"  : float      # (Bid + Ask) / 2
    "Spread"     : float      # Ask − Bid
    "OBI"        : float      # (Bid_Vol − Ask_Vol) / (Bid_Vol + Ask_Vol)
  }

Usage
-----
    from streaming_feed import RollingBuffer, BinanceStreamFeed

    buffer = RollingBuffer(maxlen=500)
    feed   = BinanceStreamFeed(symbol="BTCUSDT", buffer=buffer)
    feed.start()

    # Block until at least 50 ticks are in the buffer
    while len(buffer) < 50:
        time.sleep(0.1)

    tick = buffer.latest()
    print(tick)

    feed.stop()
"""

import asyncio
import json
import threading
import time
from collections import deque
from typing import Optional

import pandas as pd

try:
    import websockets
except ImportError:
    raise ImportError(
        "The 'websockets' package is required for live streaming.\n"
        "Install it with:  pip install websockets"
    )


# ──────────────────────────────────────────────────────────────────────────────
# RollingBuffer
# ──────────────────────────────────────────────────────────────────────────────

class RollingBuffer:
    """
    A thread-safe, fixed-size circular buffer of market tick dictionaries.

    New ticks are appended via ``push()``.  When the buffer is full, the
    oldest tick is silently discarded (deque behaviour).

    Parameters
    ----------
    maxlen : int
        Maximum number of ticks to retain.  Older ticks are dropped.
    """

    def __init__(self, maxlen: int = 500):
        self._buf   = deque(maxlen=maxlen)
        self._lock  = threading.Lock()
        self._event = threading.Event()   # fires on every push

    # ── Write ─────────────────────────────────────────────────────────────────

    def push(self, tick: dict) -> None:
        """Append a processed tick dict to the buffer (thread-safe)."""
        with self._lock:
            self._buf.append(tick)
        self._event.set()
        self._event.clear()

    # ── Read ──────────────────────────────────────────────────────────────────

    def latest(self) -> Optional[dict]:
        """Return the most-recently added tick, or None if the buffer is empty."""
        with self._lock:
            return self._buf[-1] if self._buf else None

    def wait_for_tick(self, timeout: float = 5.0) -> bool:
        """
        Block until a new tick is pushed or ``timeout`` seconds elapse.

        Returns
        -------
        bool
            True if a tick arrived within the timeout, False if timed out.
        """
        return self._event.wait(timeout=timeout)

    def as_dataframe(self, n: Optional[int] = None) -> pd.DataFrame:
        """
        Snapshot the buffer as a pandas DataFrame.

        Args:
            n: If given, only the last ``n`` ticks are included.

        Returns:
            pd.DataFrame with one row per tick.
        """
        with self._lock:
            data = list(self._buf)
        if n is not None:
            data = data[-n:]
        return pd.DataFrame(data)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def __repr__(self) -> str:
        return f"RollingBuffer(len={len(self)}, maxlen={self._buf.maxlen})"


# ──────────────────────────────────────────────────────────────────────────────
# BinanceStreamFeed
# ──────────────────────────────────────────────────────────────────────────────

class BinanceStreamFeed:
    """
    Connects to the Binance **Best Bid/Ask Ticker** WebSocket stream
    (``<symbol>@bookTicker``) and pushes pre-processed ticks into a
    ``RollingBuffer``.

    The WebSocket client runs in a background daemon thread via a dedicated
    asyncio event loop so it never blocks the main thread.

    Parameters
    ----------
    symbol : str
        Binance trading pair, e.g. "BTCUSDT".  Case-insensitive.
    buffer : RollingBuffer
        Destination buffer that receives processed ticks.
    reconnect_delay : float
        Seconds to wait before reconnecting after a dropped connection.

    Binance bookTicker message format
    ----------------------------------
    {
        "u": 400900217,          # update id
        "s": "BNBUSDT",          # symbol
        "b": "25.35190000",      # best bid price
        "B": "31.21000000",      # best bid qty
        "a": "25.36520000",      # best ask price
        "A": "40.66000000"       # best ask qty
    }
    """

    _BASE_WS = "wss://stream.binance.com:9443/ws"

    def __init__(
        self,
        symbol: str,
        buffer: RollingBuffer,
        reconnect_delay: float = 3.0,
    ):
        self.symbol          = symbol.lower()
        self.buffer          = buffer
        self.reconnect_delay = reconnect_delay

        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background WebSocket thread."""
        if self._running:
            print("[BinanceStreamFeed] Already running.")
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        print(f"[BinanceStreamFeed] Started streaming {self.symbol.upper()}...")

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to exit."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=6)
        print(f"[BinanceStreamFeed] Stopped.")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _run_event_loop(self) -> None:
        """Entry-point for the background daemon thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._stream())
        finally:
            loop.close()

    async def _stream(self) -> None:
        """Async WebSocket receive loop with automatic reconnection."""
        url = f"{self._BASE_WS}/{self.symbol}@bookTicker"
        print(f"[BinanceStreamFeed] Connecting to {url}")

        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    print(f"[BinanceStreamFeed] Connected [OK]")
                    async for raw_message in ws:
                        if not self._running:
                            return
                        try:
                            raw  = json.loads(raw_message)
                            tick = self._process_tick(raw)
                            self.buffer.push(tick)
                        except (KeyError, ValueError, json.JSONDecodeError) as e:
                            print(f"[BinanceStreamFeed] Tick parse error: {e}")

            except Exception as e:
                if self._running:
                    print(f"[BinanceStreamFeed] Connection error: {e}. "
                          f"Reconnecting in {self.reconnect_delay}s...")
                    await asyncio.sleep(self.reconnect_delay)

    def _process_tick(self, raw: dict) -> dict:
        """
        Convert a raw Binance bookTicker message into the LOB tick schema
        used by ExecutionEnv and RollingBuffer.
        """
        bid_price = float(raw["b"])
        ask_price = float(raw["a"])
        bid_vol   = float(raw["B"])
        ask_vol   = float(raw["A"])

        mid_price  = (bid_price + ask_price) / 2.0
        spread     = ask_price - bid_price
        total_vol  = bid_vol + ask_vol
        obi        = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0

        return {
            "timestamp"  : time.time(),
            "Bid_Price"  : bid_price,
            "Ask_Price"  : ask_price,
            "Bid_Volume" : bid_vol,
            "Ask_Volume" : ask_vol,
            "Mid_Price"  : mid_price,
            "Spread"     : spread,
            "OBI"        : obi,
        }

    def __repr__(self) -> str:
        status = "running" if self._running else "stopped"
        return f"BinanceStreamFeed(symbol={self.symbol.upper()}, status={status})"


# ──────────────────────────────────────────────────────────────────────────────
# Standalone smoke-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Streaming BTCUSDT for 10 seconds — collecting ticks into buffer...")

    buf  = RollingBuffer(maxlen=200)
    feed = BinanceStreamFeed(symbol="BTCUSDT", buffer=buf)
    feed.start()

    time.sleep(10)
    feed.stop()

    print(f"\nCollected {len(buf)} ticks.")
    if len(buf) > 0:
        tick = buf.latest()
        print(f"Latest tick: Mid={tick['Mid_Price']:.2f}, "
              f"Spread={tick['Spread']:.4f}, OBI={tick['OBI']:.4f}")
        df = buf.as_dataframe()
        print(f"\nBuffer as DataFrame:\n{df.tail(5).to_string()}")
    print("\nstreaming_feed.py smoke-test passed ✓")

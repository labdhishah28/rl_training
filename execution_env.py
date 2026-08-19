import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional

class ExecutionEnv(gym.Env):
    """
    Custom Environment for Optimal Trade Execution.
    The agent must liquidate (or acquire) a target inventory over a fixed time horizon.

    Supports two data modes:
        Static mode  (default): Reads pre-loaded historical arrays.  Used for all
                                 training and back-testing.
        Live mode              : Reads the latest tick from a ``RollingBuffer``
                                 pushed by ``BinanceStreamFeed``.  Used for
                                 paper-trading and live inference only.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data: pd.DataFrame, initial_inventory: float = 10000.0,
                 time_horizon: int = 100, kappa: float = 0.01, gamma: float = 0.005,
                 is_buy: bool = False, live_feed=None):
        """
        Args:
            data             : LOB DataFrame (used for static training / seeding obs space).
            initial_inventory: Total units to liquidate.
            time_horizon     : Number of steps allowed to execute.
            kappa            : Temporary market impact coefficient.
            gamma            : Permanent market impact coefficient.
            is_buy           : True to buy, False to sell/liquidate.
            live_feed        : Optional ``RollingBuffer`` instance.  When supplied,
                               the env reads live tick data instead of static arrays.
                               Use ONLY for inference/paper-trading, NOT for training.
        """
        super(ExecutionEnv, self).__init__()
        
        self.data = data
        self.max_steps = len(self.data)
        self.live_feed = live_feed    # RollingBuffer | None
        
        # Convert frequently accessed pandas columns to numpy arrays for much faster indexing in the RL loop
        self.mid_prices = self.data['Mid_Price'].values
        self.spreads = self.data['Spread'].values
        self.obis = self.data['OBI'].values
        
        # Trading parameters
        self.initial_inventory = initial_inventory
        self.time_horizon = time_horizon # Number of steps allowed to execute
        self.kappa = kappa  # Temporary impact parameter
        self.gamma = gamma  # Permanent impact parameter
        self.is_buy = is_buy # True to buy, False to sell (liquidate)
        
        # Action space: fraction of remaining inventory to trade at this step.
        # Can range from 0 to 1 (execute up to 100% of remaining inventory).
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space:
        # [time_remaining_ratio, inventory_remaining_ratio, mid_price_scaled,
        #  spread_scaled, OBI, price_momentum, rolling_volatility, spread_ratio]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.time_remaining = self.time_horizon
        self.inventory = self.initial_inventory
        self.permanent_impact_acc = 0.0
        # Rolling price history for momentum and volatility features (last 5 mid prices)
        self._price_history: list = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Pick a random starting point that allows enough time steps
        max_start = self.max_steps - self.time_horizon - 1
        if max_start <= 0:
            self.current_step = 0
        else:
            self.current_step = self.np_random.integers(0, max_start)
            
        self.time_remaining = self.time_horizon
        self.inventory = self.initial_inventory
        self.permanent_impact_acc = 0.0
        self._price_history = []
        
        return self._get_obs(), {}
        
    def _get_obs(self) -> np.ndarray:
        # ── Live mode: read from the streaming buffer ─────────────────────────
        if self.live_feed is not None:
            tick = self.live_feed.latest()
            if tick is not None:
                mid = tick['Mid_Price']
                adjusted_mid = mid + self.permanent_impact_acc
                # Update rolling price history
                self._price_history.append(mid)
                if len(self._price_history) > 5:
                    self._price_history.pop(0)
                momentum   = (self._price_history[-1] / self._price_history[0] - 1.0) if len(self._price_history) > 1 else 0.0
                volatility = float(np.std(self._price_history)) / mid if len(self._price_history) > 1 else 0.0
                spread_ratio = tick['Spread'] / mid if mid > 0 else 0.0
                return np.array([
                    self.time_remaining / self.time_horizon,
                    self.inventory / self.initial_inventory,
                    adjusted_mid / 100.0,
                    tick['Spread'] / 0.05,
                    tick['OBI'],
                    momentum * 100.0,     # scale: ~[-1, 1] for typical crypto moves
                    volatility * 1000.0,  # scale: very small raw values
                    spread_ratio * 1000.0, # scale: bps-like
                ], dtype=np.float32)

        # ── Static mode: read from pre-loaded arrays (training / back-test) ──
        mid = self.mid_prices[self.current_step]
        adjusted_mid_price = mid + self.permanent_impact_acc
        # Update rolling price history
        self._price_history.append(mid)
        if len(self._price_history) > 5:
            self._price_history.pop(0)
        momentum   = (self._price_history[-1] / self._price_history[0] - 1.0) if len(self._price_history) > 1 else 0.0
        volatility = float(np.std(self._price_history)) / mid if len(self._price_history) > 1 else 0.0
        spread     = self.spreads[self.current_step]
        spread_ratio = spread / mid if mid > 0 else 0.0
        obs = np.array([
            self.time_remaining / self.time_horizon,
            self.inventory / self.initial_inventory,
            adjusted_mid_price / 100.0,   # Scale price down
            spread / 0.05,                # Scale spread up
            self.obis[self.current_step], # Already in [-1, 1]
            momentum * 100.0,             # Price momentum (5-step)
            volatility * 1000.0,          # Rolling volatility (5-step std)
            spread_ratio * 1000.0,        # Spread as fraction of mid price (bps)
        ], dtype=np.float32)
        return obs
        
    def step(self, action):
        # ── Resolve current market state ──────────────────────────────────────
        if self.live_feed is not None:
            tick = self.live_feed.latest()
            if tick is not None:
                historical_mid = tick['Mid_Price']
                spread         = tick['Spread']
            else:
                # Fallback to last static row if buffer is temporarily empty
                idx = min(self.current_step, self.max_steps - 1)
                historical_mid = self.mid_prices[idx]
                spread         = self.spreads[idx]
        else:
            historical_mid = self.mid_prices[self.current_step]
            spread         = self.spreads[self.current_step]

        adjusted_mid = historical_mid + self.permanent_impact_acc
        
        # Action is interpreted as the fraction of the current inventory to trade
        # Force it between 0 and 1 just in case
        fraction_to_trade = float(min(max(action[0], 0.0), 1.0))
        
        # Calculate execution volume
        volume_to_trade = self.inventory * fraction_to_trade
        
        # We must execute all remaining inventory on the last step
        if self.time_remaining == 1:
            volume_to_trade = self.inventory
            
        # Execute trade
        # 1. Temporary Impact: h(v) = kappa * v
        # Assuming v here is the volume traded in this step. 
        temp_impact = self.kappa * volume_to_trade
        
        # Baseline execution price is mid price +/- half spread
        sign = 1 if self.is_buy else -1
        
        # Execution price: worse price due to crossing the spread and temporary impact
        exec_price = adjusted_mid + sign * (spread / 2.0) + sign * temp_impact
        
        # 2. Permanent Impact: g(v) = gamma * v
        # Affects future mid prices
        perm_impact = sign * self.gamma * volume_to_trade
        self.permanent_impact_acc += perm_impact
        
        # Update state
        cash_flow = -sign * volume_to_trade * exec_price
        self.inventory -= volume_to_trade
        self.time_remaining -= 1
        self.current_step += 1
        
        # Calculate Reward
        # For liquidation, reward is the cash flow.
        # We scale it down by 10,000 to keep the reward values closer to O(1) for PPO stability.
        reward = cash_flow / 10000.0
        
        # Holding penalty: encourages spreading execution across time steps
        # Higher value = smoother/slower execution (less front-loading)
        holding_penalty = (self.inventory * 0.005) / 10000.0
        reward -= holding_penalty
        
        # Check termination
        terminated = bool(self.time_remaining <= 0 or self.inventory <= 1e-5)
        
        # If we failed to liquidate everything on the last step
        if terminated and self.inventory > 1e-5:
            # Reduced penalty for unexecuted inventory (from 1.5x to 0.5x) to prevent gradient explosion
            penalty = self.inventory * (adjusted_mid + sign * spread) * 0.5 
            reward -= (penalty / 10000.0)
            self.inventory = 0.0 # Force zero
            
        truncated = False
        
        info = {
            'execution_price': exec_price,
            'volume_traded': volume_to_trade,
            'inventory_remaining': self.inventory,
            'temporary_impact': temp_impact,
            'permanent_impact_acc': self.permanent_impact_acc
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        # We can implement a rendering logic later if needed
        pass

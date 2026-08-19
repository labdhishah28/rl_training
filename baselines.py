import numpy as np

class BaselineModels:
    """
    Implements TWAP, VWAP, and Almgren-Chriss (AC) analytical baselines.
    These models calculate the target remaining inventory at each timestep.
    """
    
    @staticmethod
    def get_twap_trajectory(initial_inventory: float, time_horizon: int) -> np.ndarray:
        """
        Calculates the Time-Weighted Average Price (TWAP) trajectory.
        Slices the inventory evenly across all time steps.
        
        Args:
            initial_inventory: The total shares to liquidate
            time_horizon: Total number of steps
            
        Returns:
            np.ndarray: An array of size (time_horizon,) representing the target 
                        remaining inventory at each step.
        """
        # Linear decay from initial_inventory down to 0
        trajectory = np.linspace(initial_inventory, 0, time_horizon + 1)
        # We drop the first element (which is just the initial inventory at t=0)
        # and return the remaining inventory targets for steps 1 through T
        return trajectory[1:]

    @staticmethod
    def get_vwap_trajectory(initial_inventory: float, volume_profile: np.ndarray) -> np.ndarray:
        """
        Calculates the Volume-Weighted Average Price (VWAP) trajectory.
        Executes inventory proportional to the historical volume profile.
        
        Args:
            initial_inventory: The total shares to liquidate
            volume_profile: An array of size (time_horizon,) representing the expected 
                            market volume at each step
                            
        Returns:
            np.ndarray: Target remaining inventory at each step.
        """
        total_market_volume = np.sum(volume_profile)
        if total_market_volume == 0:
            # Fallback to TWAP if no volume data is available
            return BaselineModels.get_twap_trajectory(initial_inventory, len(volume_profile))
            
        # Calculate cumulative proportion of volume traded up to each step
        cumulative_volume = np.cumsum(volume_profile)
        volume_proportions = cumulative_volume / total_market_volume
        
        # We start with initial_inventory and subtract the proportion we should have traded
        trajectory = initial_inventory * (1 - volume_proportions)
        return trajectory

    @staticmethod
    def get_almgren_chriss_trajectory(initial_inventory: float, time_horizon: int, 
                                      risk_aversion: float, 
                                      kappa: float, 
                                      sigma: float = 1.0) -> np.ndarray:
        """
        Calculates the optimal Almgren-Chriss trading trajectory.
        Balances market risk (price variance) against market impact.
        
        Args:
            initial_inventory (X): Total shares to liquidate
            time_horizon (T): Number of steps
            risk_aversion (lambda): Penalty for variance/market risk. Higher = trade faster.
            kappa (eta in AC): Temporary impact coefficient. Higher = trade slower.
            sigma: Volatility of the asset
            
        Returns:
            np.ndarray: Target remaining inventory at each step.
        """
        # If risk aversion is 0, the trader doesn't care about price crashing.
        # The optimal solution to minimize market impact is a straight line (TWAP).
        if risk_aversion <= 0:
            return BaselineModels.get_twap_trajectory(initial_inventory, time_horizon)
            
        # kappa_tilde in AC is sqrt((lambda * sigma^2) / eta)
        # We add a tiny epsilon to prevent division by zero in case kappa is 0
        kappa_tilde = np.sqrt((risk_aversion * (sigma ** 2)) / (kappa + 1e-8))
        
        T = time_horizon
        t_array = np.arange(1, T + 1)
        
        numerator = np.sinh(kappa_tilde * (T - t_array))
        denominator = np.sinh(kappa_tilde * T)
        
        trajectory = initial_inventory * (numerator / denominator)
            
        return trajectory

if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt
    
    T = 100
    X = 10000
    
    twap = BaselineModels.get_twap_trajectory(X, T)
    ac_low_risk = BaselineModels.get_almgren_chriss_trajectory(X, T, risk_aversion=1e-6, kappa=0.01)
    ac_high_risk = BaselineModels.get_almgren_chriss_trajectory(X, T, risk_aversion=1e-4, kappa=0.01)
    
    print("Baseline Models module loaded successfully.")
    print(f"TWAP Final State: {twap[-1]}")
    print(f"AC Low Risk Final State: {ac_low_risk[-1]}")

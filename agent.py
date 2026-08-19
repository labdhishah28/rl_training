import os
from stable_baselines3 import PPO
import warnings

# Suppress some common stable-baselines3 warnings for clean output
warnings.filterwarnings("ignore")

class ExecutionAgent:
    """
    Wrapper for the Stable-Baselines3 Deep RL Agent.
    Uses PPO (Proximal Policy Optimization) or RecurrentPPO (LSTM).
    """
    def __init__(self, env, model_path="ppo_execution_agent", tensorboard_log=None, **kwargs):
        self.env = env
        self.model_path = model_path
        self.tensorboard_log = tensorboard_log
        
        # Merge default kwargs with provided kwargs
        model_kwargs = {"verbose": 1, "learning_rate": 3e-4, "tensorboard_log": self.tensorboard_log}
        model_kwargs.update(kwargs)
        
        # Initialize PPO
        self.model = PPO("MlpPolicy", self.env, **model_kwargs)

    def train(self, total_timesteps: int = 10000, callback=None):
        """
        Trains the RL agent in the Gym environment.
        """
        print(f"Starting PPO training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        print("Training completed.")
        
    def save(self):
        """
        Saves the trained neural network weights.
        """
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}.zip")

    def load(self):
        """
        Loads the trained neural network weights from disk.
        """
        if os.path.exists(f"{self.model_path}.zip"):
            self.model = PPO.load(self.model_path, env=self.env)
            print(f"Model loaded from {self.model_path}.zip")
        else:
            print(f"Model file {self.model_path}.zip not found. Please train first.")

    def predict(self, observation, deterministic=True):
        """
        Given a state observation vector, asks the Actor network for the optimal action.
        deterministic=True means no exploration noise is added (use this for testing/live).
        """
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action, None

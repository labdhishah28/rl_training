import optuna
import pandas as pd
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from execution_env import ExecutionEnv
from agent import ExecutionAgent
from loader import DataLoader
import warnings
warnings.filterwarnings("ignore")

# Load data once
data_path = "dummy_market_data.csv"
loader = DataLoader(data_path)
df = loader.load_data()

time_horizon = 100
initial_inventory = 10000.0

def make_env():
    return ExecutionEnv(
        data=df, 
        initial_inventory=initial_inventory, 
        time_horizon=time_horizon, 
        kappa=0.01, 
        gamma=0.005, 
        is_buy=False
    )

def optimize_ppo(trial):
    """
    Optuna objective function for PPO hyperparameters.
    """
    # 1. Propose hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.0000001, 0.1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    
    # 2. Setup Vectorized Environment
    vec_env = make_vec_env(make_env, n_envs=4) # Use 4 parallel envs for speed
    
    # 3. Instantiate the agent with proposed parameters
    # Note: verbose=0 to avoid spamming the console during 10 trials
    agent = ExecutionAgent(
        env=vec_env, 
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        batch_size=batch_size,
        gamma=gamma,
        n_steps=n_steps,
        verbose=0
    )
    
    # 4. Train the agent (keep timesteps low for fast optimization)
    # 20,000 steps is enough to see which parameters converge faster
    agent.train(total_timesteps=20000)
    
    # 5. Evaluate the agent
    eval_env = make_env()
    mean_reward, std_reward = evaluate_policy(agent.model, eval_env, n_eval_episodes=5, deterministic=True)
    
    return mean_reward

if __name__ == "__main__":
    print("Starting Optuna Hyperparameter Optimization...")
    # Maximize the reward (Cash Flow)
    study = optuna.create_study(direction="maximize")
    
    # Run 10 trials to find the best settings quickly (you can increase this later)
    study.optimize(optimize_ppo, n_trials=10)
    
    print("\n========================================")
    print(" Optuna Optimization Complete ")
    print("========================================")
    print(f"Best Trial Reward: {study.best_trial.value}")
    print("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    print("\nYou can now plug these exact values into train.py!")

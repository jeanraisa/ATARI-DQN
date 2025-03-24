import gymnasium as gym
import os
import csv
import yaml
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

# Create necessary directories
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("best_model", exist_ok=True)

# Hyperparameter Table (for documentation)
hyperparameter_table = []

class RewardLoggingCallback(BaseCallback):
    """Custom callback for logging rewards."""
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0  # Track episode reward manually

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]  # Extract reward
        self.current_episode_reward += reward

        if "dones" in self.locals and self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(len(self.episode_rewards))
            self.current_episode_reward = 0  # Reset for next episode
        return True


def evaluate_model(model, env, num_episodes=10):
    """Evaluate the trained model over multiple episodes."""
    import numpy as np  # Ensure NumPy is imported
    total_rewards = []
    total_lengths = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):  # Handle different Gym versions
            obs = obs[0]  # Extract only observation
            
        done, ep_reward, ep_length = False, 0, 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            
            if len(step_result) == 4:
                obs, reward, done, info = step_result
                truncated = False
            else:
                obs, reward, done, truncated, info = step_result

            ep_reward += reward[0]
            ep_length += 1
            done = done[0] if isinstance(done, (list, tuple, np.ndarray)) else done  # Fix handling

        total_rewards.append(ep_reward)
        total_lengths.append(ep_length)
    
    avg_reward = sum(total_rewards) / num_episodes
    avg_ep_length = sum(total_lengths) / num_episodes
    return avg_reward, avg_ep_length



def train_dqn(env, policy, config):
    """Train the model and return performance metrics."""
    set_random_seed(config["training"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DQN(
        policy, env,
        learning_rate=config["model"]["lr"],
        gamma=config["training"]["gamma"],
        batch_size=config["training"]["batch_size"],
        exploration_fraction=config["epsilon"]["fraction"],
        exploration_final_eps=config["epsilon"]["end"],
        buffer_size=config["model"].get("buffer_size", 100000),
        verbose=1, tensorboard_log="./logs/dqn_tensorboard/",
        device=device
    )
    
    callbacks = [
        CheckpointCallback(save_freq=10000, save_path='./checkpoints/', name_prefix='dqn_model'),
        EvalCallback(env, best_model_save_path='./best_model/', log_path='./logs/eval_log/', eval_freq=5000),
        RewardLoggingCallback()
    ]

    model.learn(total_timesteps=config["training"]["num_timesteps"], callback=callbacks)

    # Evaluate the trained model
    avg_reward, avg_ep_length = evaluate_model(model, env)

    # Save the trained model
    model.save("dqn_model.zip")

    return model, avg_reward, avg_ep_length


def log_hyperparameters():
    """Save hyperparameter tuning results to a CSV file."""
    with open("logs/hyperparameter_tuning_results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Policy", "Learning Rate", "Gamma", "Batch Size",
            "Epsilon Start", "Epsilon End", "Epsilon Decay", "Noted Behavior"
        ])
        
        for row in hyperparameter_table:
            writer.writerow([
                row["Policy"], row["Learning Rate"], row["Gamma"], row["Batch Size"],
                row["Epsilon Start"], row["Epsilon End"], row["Epsilon Decay"], row["Noted Behavior"]
            ])


if __name__ == "__main__":
    # Load configuration from config.yaml
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    required_keys = ["env", "training", "model", "epsilon"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")

    # Setup environment
    env = DummyVecEnv([lambda: Monitor(gym.make(config["env"]))])  # Wrapped with Monitor

    # Define model policies
    policies = ["MlpPolicy", "CnnPolicy"]

    # Train DQN with different policies
    for policy in policies:
        print(f"\nTraining with {policy} policy...")
        model, avg_reward, avg_ep_length = train_dqn(env, policy, config)

        # Log hyperparameters and behavior for the experiment
        hyperparameter_table.append({
            "Policy": policy,
            "Learning Rate": config["model"]["lr"],
            "Gamma": config["training"]["gamma"],
            "Batch Size": config["training"]["batch_size"],
            "Epsilon Start": config["epsilon"]["start"],
            "Epsilon End": config["epsilon"]["end"],
            "Epsilon Decay": config["epsilon"].get("decay", "N/A"),
            "Noted Behavior": f"Avg Reward: {float(avg_reward):.2f}, Avg Episode Length: {float(avg_ep_length):.1f}"
        })

    # Save hyperparameter tuning results
    log_hyperparameters()
    print("Training completed. Models saved as dqn_model.zip.")
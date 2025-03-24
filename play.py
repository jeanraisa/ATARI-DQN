# !pip install sb3-contrib #installing required module
import gymnasium as gym
import numpy as np
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy
import time
import os
import cv2
import glob
from datetime import datetime
from gymnasium.wrappers import RecordVideo
import ale_py  # import the ale-py library
# Fix: Import GreedyQPolicy from the correct location
#from sb3_contrib.common.policies import GreedyQPolicy  
from stable_baselines3.common.policies import BasePolicy #changed the import to the original location
#from sb3_contrib.q_learning import GreedyQPolicy  

def make_atari_env(env_id):
    """
    Create an Atari environment with proper wrappers.
    Ensures RGB observations instead of grayscale.
    """
    # Set render_mode to "rgb_array" for RecordVideo
    env = gym.make(env_id, render_mode="rgb_array")
    env = AtariWrapper(env)
    return env


def preprocess_observation(obs):
    """
    Ensures the observation is in the correct format (3, 210, 160).
    """
    if obs.shape[-1] == 1:  # If grayscale (84, 84, 1)
        obs = np.repeat(obs, 3, axis=-1)  # Convert to (84, 84, 3)
    obs = cv2.resize(obs, (160, 210))  # Resize to (210, 160, 3)
    obs = np.transpose(obs, (2, 0, 1))  # Convert to (3, 210, 160)
    return obs


def record_video(env_name, model, num_episodes=10):
    """
    Records gameplay video of the trained agent.
    """
    # Create video directory
    video_dir = f"./videos/{env_name.split('/')[-1]}{datetime.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(video_dir, exist_ok=True)

    # Create environment with RecordVideo wrapper
    env = make_atari_env(env_name)  # removed render_mode as env will not be rendered
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda x: True,  # Record every episode
        name_prefix=f"{env_name.split('/')[-1]}"
    )

    rewards = []
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done, truncated = False, False

        while not (done or truncated):
            obs = preprocess_observation(obs)  # Ensure correct shape
            # Use GreedyQPolicy to select actions greedily
            action, _states = model.predict(obs, deterministic=True)  

            # The predict method is returning a scalar, so no need to index it.
            # If the predict method starts returning an array, this might need to be changed back
            # action = action[0]  # Extract action value from array

            if action >= env.action_space.n:  # Clip action if out of range
                action = env.action_space.n - 1
            elif action < 0:
                action = 0

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
        print(f"Episode {episode+1} reward: {episode_reward}")

    env.close()
    print(f"Videos saved to {video_dir}")
    return rewards, video_dir


def merge_videos(video_dir, output_filename="merged_video.mp4", fps=30):
    """
    Merges all videos in the given directory into a single video.
    """
    video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not video_files:
        print("No videos found to merge.")
        return None

    # Get video properties from the first video
    first_video = cv2.VideoCapture(video_files[0])
    frame_width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = os.path.join(video_dir, output_filename)

    # Initialize VideoWriter
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()
    print(f"Merged video saved to: {output_path}")
    return output_path


def video_to_gif(video_path, gif_path, start_time=0, end_time=None, fps=10):
    """
    Converts the video to a GIF.
    """
    from moviepy.editor import VideoFileClip

    video = VideoFileClip(video_path)
    if end_time is not None:
        video = video.subclip(start_time, end_time)
    video.write_gif(gif_path, fps=fps)
    print(f"GIF saved at {gif_path}")

# Change _name_ to __name__
if __name__ == "__main__":
    # Load the trained model
    model_path = "/content/dqn_model (1).zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = DQN.load(model_path, buffer_size=10000)
    print(f"Model loaded from {model_path}")

    # Set the policy to GreedyQPolicy
    #model.policy = GreedyQPolicy(model.policy) #this line is likely not necessary

    # Environment setup
    env_name = "ALE/Breakout-v5"

    # Record video
    print("\n=== Recording gameplay videos ===")
    video_rewards, video_dir = record_video(env_name, model, num_episodes=10)

    # Merge recorded videos
    print("\n=== Merging recorded videos ===")
    merged_video_path = merge_videos(video_dir)
    if merged_video_path:
        print(f"Merged video available at: {merged_video_path}")

    # Convert the merged video to a GIF
    if merged_video_path:
        gif_path = os.path.join(video_dir, "gameplay.gif")
        video_to_gif(merged_video_path, gif_path, fps=15)
        print(f"GIF saved at: {gif_path}")

    # Display performance summary
    print("\n=== Performance Summary ===")
    print(f"Average reward (video): {np.mean(video_rewards):.2f}")
    print(f"Videos saved to: {video_dir}")
    if merged_video_path:
        print(f"Merged video located at: {merged_video_path}")
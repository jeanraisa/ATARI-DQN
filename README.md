
# DEMO






![gameplay](https://github.com/user-attachments/assets/2750ab15-1b0a-4c70-8e53-ecbf9eda6bfe)


![gameplay](https://github.com/jeanraisa/ATARI-DQN/blob/main/videos/merged_video%20(1)%20true.mp4)



# DQN Agent for Atari Breakout

This project implements a Deep Q-Network (DQN) agent to play Atari's Breakout game using Stable-Baselines3 and Gymnasium. The implementation includes both training and evaluation scripts.

# Project Structure

train.py: Script for training the DQN agent

play.py: Script for running the trained agent

requirements.txt: List of required packages

# Prerequisites

System Requirements

Python 3.8 or higher

pip (Python package installer)

Virtual environment (recommended)

Required Python Packages

gymnasium[atari]

stable-baselines3[extra]

ale-py

Installation

Clone this repository:

git clone https://github.com/jeanraisa/ATARI-DQN.git
cd ATARI-DQN

Create and activate a virtual environment (recommended):

On Windows

python -m venv venv
venv\Scripts\activate

On macOS/Linux

python3 -m venv venv
source venv/bin/activate

Install required packages:

pip install gymnasium[atari] stable-baselines3[extra] ale-py

# Usage

Training the agent:

python train.py

Playing with trained agent:

python play.py

# Hyperparameter Tuning

![Table](https://github.com/jeanraisa/ATARI-DQN/blob/main/Table/table1.png)


# 1. Learning Rate (lr=0.0005)
The learning rate controls how much the model updates its weights during training.

Impact:

A lower learning rate (e.g., 0.0001) leads to slower learning but more stability.

A higher learning rate (e.g., 0.01) can speed up learning but may cause instability.

0.0005 is a reasonable balance, allowing steady convergence without drastic updates.

# 2. Discount Factor (gamma=0.97)
Gamma (γ) determines how much future rewards matter compared to immediate rewards.

Impact:

If γ = 0, the agent only cares about immediate rewards (myopic behavior).

If γ = 1, the agent considers all future rewards equally (long-term planning).

0.97 gives high importance to future rewards while still valuing immediate rewards. This helps the agent make smarter, long-term decisions.

# 3. Batch Size (batch_size=64)
The number of past experiences sampled from the replay buffer for each update.

Impact:

Smaller batches (e.g., 32) lead to more frequent updates but may result in high variance.

Larger batches (e.g., 128 or 256) provide more stable updates but require more memory and slow down training.

64 is a common choice, balancing efficiency and stability.

# 4. Exploration-Exploitation Tradeoff (Epsilon Parameters)
Epsilon (ε) governs the balance between exploration (trying new actions) and exploitation (using learned actions).

Epsilon Start (epsilon_start=1)

Starts with pure exploration (100% random actions).

Ensures the agent gathers diverse experiences early in training.

Epsilon End (epsilon_end=0.1)

The final probability of taking a random action.

Ensures the agent still explores a little after training.

Epsilon Decay (epsilon_decay=20000)

Controls how fast the agent shifts from exploration to exploitation.

Higher decay (e.g., 50,000) → Exploration lasts longer.

Lower decay (e.g., 5,000) → Exploitation starts too soon.

20000 is a moderate setting, meaning the agent gradually transitions to exploitation.




# Discussion on Results

The results indicate varying performance of the trained agent across different evaluation settings. Below is an analysis of each result:

# 1. Avg Reward: 650.00, Avg Episode Length: 1439.0

This result suggests a well-performing agent capable of surviving for a long time while maximizing rewards.

A long episode length (1439 steps) indicates that the agent makes informed decisions, allowing it to stay in the game longer.

The high reward suggests that the policy is well-optimized and effective.

# 2. Avg Reward: 9.40, Avg Episode Length: 9.4

The agent is performing poorly, with very short episodes.

Possible reasons:

The model is undertrained.

The agent fails early due to suboptimal actions.

Potential issues with the exploration-exploitation balance.

Further training or fine-tuning of hyperparameters (e.g., learning rate, epsilon decay) may be needed.

# 3. Avg Reward: 200.00, Avg Episode Length: 572.0

This is a moderate performance—better than (2) and (4) but not as strong as (1).

The agent is able to sustain itself for a reasonable number of steps (572) and accumulate rewards.

The model has learned some useful policies but may require further training for improvement.

# 4. Avg Reward: 9.50, Avg Episode Length: 9.5

Similar to (2), this result suggests an underperforming agent.

The short episode length indicates that the agent is failing early, possibly due to poor decision-making.

The model might not be well-trained, or it could be stuck in a suboptimal policy.

# Key Takeaways & Recommendations

Strong performance (Result 1) indicates a well-trained model with effective policy learning.

Very low performance (Results 2 & 4) suggests either poor training, lack of exploration, or failure in learning a stable policy.

Moderate performance (Result 3) implies partial learning, but further optimization (e.g., more training steps, adjusting the reward function) is needed.

# Contribution

1.  Dohessienkan Xavier Gnondoyi: Training and Model development( Developed the train script)
2. Irakoze Jean Raisa: Developed the play script for Gameplay and evaluation and README Documentation
3. Cynthia Nekesa: READMe Documentation and play script development

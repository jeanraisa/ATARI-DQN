# DQN Agent for Atari Breakout

This project implements a Deep Q-Network (DQN) agent to play Atari's Breakout game using Stable-Baselines3 and Gymnasium. The implementation includes both training and evaluation scripts.

# Project Structure

* `train.py`: Script for training the DQN agent
* `play.py`: Script for running the trained agent
* `requirements.txt`: List of required packages

# Prerequisites

# System Requirements

  * Python 3.8 or higher
  * pip (Python package installer)
  * Virtual environment (recommended)

# Required Python Packages

  * gymnasium[atari]
  * stable-baselines3[extra]
  * ale-py

# Installation

  1. Clone this repository:

  `git clone https://github.com/jeanraisa/ATARI-DQN.git
  cd ATARI-DQN`

  2. Create and activate a virtual environment (recommended):

  # On Windows
  `python -m venv venv
  venv\Scripts\activate`

 # On macOS/Linux
 `python3 -m venv venv
 source venv/bin/activate`

  3. Install required packages:

     `pip install gymnasium[atari] stable-baselines3[extra] ale-py`

# Usage

Training the agent:

`python train.py`

Playing with trained agent:

`python play.py`

# Training Configuration

* Learning rate: 
* Buffer size: 
* Batch size: 32
* Gamma: 
* Exploration fraction: 0.1
* Training steps: 

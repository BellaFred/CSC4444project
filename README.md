# Spades AI Agent (CSC 4444 Project)

Nuha Syed, Bella Frederick, Myles Guidry, Makayla Files

This project focuses on developing an AI agent for the card game Spades to create an agent that makes accurate bidding decisions, plays cards strategically during trick-taking, and improves its performance through learning and simulation. To achieve this, we implement a complete Spades game environment, design an AI agent using Monte Carlo Tree Search (MCTS), applying reinforcement learning through self-play to enhance decision-making over time, and evaluate the agent’s performance against baseline approaches such as a greedy agent.

## Methodology

### Game Environment
Simulates full Spades gameplay:
- Card dealing
- Bidding phase
- Trick-taking phase

Maintains:
- Game state
- Player actions
- Reward signals

### AI Approach

#### Monte Carlo Tree Search (MCTS)
- Explores possible future game states through simulations
- Uses rollouts to estimate action values
- Helps guide both bidding and card selection

#### Reinforcement Learning (Self-Play)
- Agent trains by playing against itself
- Learns from reward feedback after each round/game
- Gradually improves policy over time

### State Representation
- Player’s current hand
- Cards already played
- Bids from all players
- Game history

### Action Space
- Bidding phase: choose number of tricks to bid
- Play phase: select a valid card to play each turn

## System Design & Tools
- Language: Python
- Framework: TinyZero
- Weights & Biases (wandb) for experiment tracking
- Algorithms:
  - Monte Carlo Tree Search
  - Reinforcement Learning

## Experimental Setup
- Initial testing performed on a simple Tic-Tac-Toe environment
- Used to validate training pipeline and run locally (offline mode)
- Transition planned to full Spades environment

## Results
-

## Setup/Running the Project

1. Clone the Repository
-git clone https://github.com/BellaFred/CSC4444project.git
-cd <project-folder-name>

2. Install Requirements
-Python 3.8+
-pip install -r requirements.txt
If there is no requirements file, install manually:
-pip install numpy torch (wandb)

3. Set Up Weights & Biases (wandb)
Used for tracking experiments, but it is not required to run locally.
-pip install wandb
-wandb login
   -You may need to create an account at Weights & Biases (https://wandb.ai)

4. Run the Project
Run Training
-python train.py
Run a Test Game
-python play.py

5. Running Example (TinyZero)
To test the framework before using Spades:
-python examples/tictactoe/train.py
  -This runs a simpler game to verify everything is working and is useful for debugging before running full Spades training

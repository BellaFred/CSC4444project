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

## Setup/Running the Project

1. Clone the Repository
-git clone https://github.com/BellaFred/CSC4444project.git

2. Install Requirements  
-Python 3.14.4  
     python3 -m venv venv  
     source venv/bin/activate  
-pip install -r  
-pip install numpy torch tqdm wandb numba  

3. Set Up Weights & Biases (wandb)  
Used for tracking experiments, but it is not required to run locally.  
- pip install wandb  
  wandb login  
   - You may need to create an account at Weights & Biases (https://wandb.ai)  
   
6. Run the Project  
Run Training  
- tinyzero/spades/train.py
- To change the number of games the agent is trained on, adjust the variables SELFPLAY_GAMES and BATCH_SIZE in train.py.
- For reference, SELFPLAY_GAMES = 50 and BATCH_SIZE = 10 takes around 5 mins to run.  
- python train.py  
Run a Test Game  
- tinyzero/spades/eval.py
- To change the number of evaluation games, adjust the variable EVAL_GAMES in eval.py.
- For reference, EVAL_GAMES = 10 takes around 20 mins to run.  
- python eval.py   

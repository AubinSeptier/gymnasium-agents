# Gymnasium Agents

## Introduction

The aim of this project is to **train** and **evaluate** different agents on games using the **Gymnasium** library and exploring different Reinforcement Learning concepts and methods. There are four types of agents to play two games : **Othello** and **Connect 4**.  
The agents are:
- **Random Agent** : A simple agent that takes random actions.
- **Monte Carlo Tree Search Agent** : An agent that uses Monte Carlo Tree Search to select actions.
- **Deep Q Network Agent** : An agent that uses a Deep Q Network to select actions.
- **AlphaZero Agent** : An agent that uses a light-version of the AlphaZero algorithm to select actions.  

## ⚙️Architecture of the project

The code is written in **Python** and uses the **Pytorch** library for neural networks.   

For **Connect 4**, all codes are in `src/C4_adapter_only` :
- `Agents` :  Contains the code for the four agents (Random, MCTS, DQN, AlphaZero).
- `Env` : Contains the Connect 4 environment code.
- `Models` : Contains the trained models for the DQN and AlphaZero agents (.pt for trained model and .pkl for parameters).
- `Utils` : Contains codes to evaluate the agents and visualize a game between two agents.

For **Othello**, all codes are in `src/Othello` :
- `Agents` : Contains the code for the four agents (Random, MCTS, DQN, AlphaZero).
- `Env` : Contains the the Othello environment code.
- `Models` : Contains the trained models for the DQN and AlphaZero agents (.pt for trained model and .pkl for parameters).
- `Utils` : Contains codes to evaluate the agents and visualize a game between two agents.

To **train and evaluate the agents**, you can run the following notebooks in the `src` folder:
- `main_AZ.ipynb` : Training and evaluation of the AlphaZero agent on Othello.
- `main_c4.ipynb` : Training and evaluation of different agents (Random, MCTS, DQN, AlphaZero) on Connect 4.
- `main_Othello.ipynb` : Training and evaluation of Random, MCTS and DQN agents on Othello.


## ⚒️Installation

Here's a step-by-step guide to installing the project on your local machine:

1. Clone the repository on your local machine using the following command:
```bash
git clone https://github.com/AubinSeptier/gymnasium-agents.git
```

2. Open the project's root directory in your favorite IDE or use the terminal.

3. Create a venv or conda environment and install the necessary packages:
```bash
pip install -r requirements.txt
```

4. Then, depending on what you want to do, execute the required action:  
- If you want to train and evaluate agents on Connect 4, run the `main_c4.ipynb` notebook in the `src` folder.
- If you want to train and evaluate agents on Othello, run the `main_Othello.ipynb` notebook in the `src` folder.
- If you want to train and evaluate the AlphaZero agent on Othello, run the `main_AZ.ipynb` notebook in the `src` folder.
- Create new scripts to use the game visualizer and watch a game between two agents.
- Play Othello against one of the trained agent with `src/utils/play_human.py`.
- Etc.


## 📅What's next?

The project can be improved in several ways:

- Implement a script to play directly against an agent at Connect 4.  

- Develop other types of agents (e.g. PPO, A2C, etc.) and compare their performance.    

- Develop more complex environments and games.  
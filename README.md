# Gymnasium Agents

This repository contains the code to train and evaluate four models in the Gymnasium environment. The models are:
- Random Agent : A simple agent that takes random actions.
- Monte Carlo Tree Search Agent : An agent that uses Monte Carlo Tree Search to select actions.
- Deep Q Network Agent : An agent that uses a Deep Q Network to select actions.
- AlphaZero Agent : An agent that uses a light-version of the AlphaZero algorithm to select actions.  

The code is available to train the agents and evaluate their performance on two games : Othello and Connect 4. 

## Architecture of the project

The code is written in Python and uses the Pytorch library for the neural networks.   

For Connect 4, all codes are in `src/C4_adapter_only` :
- `Agents` :  Contains the code for the four agents (Random, MCTS, DQN, AlphaZero).
- `Env` : Contains the code for the Connect 4 environment.
- `Models` : Contains the trained models for the DQN and AlphaZero agents (.pt and .pkl files).
- `Utils` : Contains the code to evaluate the agents and visualize a game between two agents.

For Othello, all codes are in `src/Othello` :
- `Agents` : Contains the code for the four agents (Random, MCTS, DQN, AlphaZero).
- `Env` : Contains the code for the Othello environment.
- `Models` : Contains the trained models for the DQN and AlphaZero agents (.pt and .pkl files).
- `Utils` : Contains the code to evaluate the agents and visualize a game between two agents.

To train and evaluate the agents, you can run the following notebooks in the `src` folder:
- `main_AZ.ipynb` : Train and evaluate the AlphaZero agent on Othello.
- `main_c4.ipynb` : Train and evaluate the different agents on Connect 4.
- `main_Othello.ipynb` : Train and evaluate the MCTS and DQN agents on Othello

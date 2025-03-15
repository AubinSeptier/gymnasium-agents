import sys
sys.path.append('./src')  
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from Agents.alpha_zero.model import SimpleAlphaZeroNet
from Agents.alpha_zero.mcts_alpha_zero import MCTSAgent
from Env.env import OthelloEnv
from Agents.alpha_zero.self_play import self_play_game  # Now returns (training_examples, final_reward)


def train_loop():
    env = OthelloEnv()
    network = SimpleAlphaZeroNet()
    agent = MCTSAgent(network, num_simulations=100)
    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    num_iterations = 1000
    losses = []    # List to store loss values
    rewards = []   # List to store total reward for each game

    for iter in range(num_iterations):
        examples, final_reward = self_play_game(agent, env)
        rewards.append(final_reward)
        
        states = torch.cat([ex[0] for ex in examples], dim=0)
        target_policies = torch.tensor(np.array([ex[1] for ex in examples]), dtype=torch.float32)
        target_values = torch.tensor(np.array([ex[2] for ex in examples]), dtype=torch.float32).unsqueeze(1)
        
        network_output = network(states)
        pred_policy_logits = network_output.policy_logits
        pred_value = network_output.value
        
        loss_value = nn.MSELoss()(pred_value, target_values)
        loss_policy = nn.CrossEntropyLoss()(pred_policy_logits, torch.argmax(target_policies, dim=1))
        loss = loss_value + loss_policy
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"Iteration {iter+1}/{num_iterations}, Loss: {loss.item():.4f}, Final Reward: {final_reward:.2f}")
    
    return losses, rewards

if __name__ == "__main__":
    training_losses, training_rewards = train_loop()

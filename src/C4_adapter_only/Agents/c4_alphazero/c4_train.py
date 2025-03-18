import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from C4_adapter_only.Agents.c4_alphazero.c4_model import ConnectFourAlphaZeroNet
from C4_adapter_only.Agents.c4_alphazero.c4_az_agent import ConnectFourAlphaZeroAgent
from C4_adapter_only.Agents.c4_alphazero.c4_self_play import self_play_game
from C4_adapter_only.Env.c4_env import ConnectFourEnv


def train_loop(num_iterations=200, num_episodes=50, learning_rate=1e-3):
    env = ConnectFourEnv()
    network = ConnectFourAlphaZeroNet()
    agent = ConnectFourAlphaZeroAgent(network, num_simulations=100)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    
    losses = []
    rewards = []
    
    for iter in range(num_iterations):
        examples, final_reward = self_play_game(agent, env)
        rewards.append(final_reward)

        # Validate examples before processing
        valid_examples = []
        for i, ex in enumerate(examples):
            if isinstance(ex[1], (list, np.ndarray)):
                valid_examples.append(ex)
            else:
                print(f"Warning: Invalid policy at index {i}: {ex[1]} (Type: {type(ex[1])})")

        if not valid_examples:
            print(f"Skipping iteration {iter+1} due to invalid data.")
            continue

        states = torch.cat([ex[0].unsqueeze(1) if ex[0].shape[1] != 1 else ex[0] for ex in valid_examples], dim=0)
        target_policies = torch.tensor(np.array([ex[1] for ex in valid_examples], dtype=np.float32))  # Ensuring valid dtype
        target_values = torch.tensor(np.array([ex[2] for ex in valid_examples], dtype=np.float32)).unsqueeze(1)
        
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

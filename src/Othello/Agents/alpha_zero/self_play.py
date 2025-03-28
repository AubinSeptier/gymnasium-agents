import torch
from Othello.Env.env import OthelloEnv
from Agents.alpha_zero.mcts_alpha_zero import MCTSAgent

def self_play_game(agent: MCTSAgent, env: OthelloEnv):
    states = []
    policies = []
    players = []
    rewards = []  

    obs, _ = env.reset()
    done = False
    while not done:
        board = obs["board"]
        state_input = torch.tensor(board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        action, policy = agent.choose_action_and_policy(env)
        states.append(state_input)
        policies.append(policy)
        players.append(obs["current_player"])
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated

    # Determine outcome based on final score
    black_count, white_count = env._get_score()
    if black_count > white_count:
        outcome = 1.0
    elif white_count > black_count:
        outcome = -1.0
    else:
        outcome = 0.0

    training_examples = []
    for state, policy, player in zip(states, policies, players):
        value_target = outcome if player == 0 else -outcome
        training_examples.append((state, policy, value_target))

    total_reward = sum(rewards)
    return training_examples, total_reward

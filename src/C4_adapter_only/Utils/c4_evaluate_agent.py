from tqdm import tqdm
from C4_adapter_only.Env.c4_env import ConnectFourEnv


def evaluate_connect_four_agent(agent, opponent, num_games=100, render=False):
    """Evaluates an agent against an opponent over multiple games."""
    env = ConnectFourEnv()
    wins = 0
    losses = 0
    draws = 0
    total_rewards = 0
    
    for game in tqdm(range(num_games), desc=f"{agent.name} vs {opponent.name}"):
        # Reset the environment
        obs, _ = env.reset()
        done = False
        game_reward = 0
        
        # Play the game
        while not done:
            # Determine which agent plays (RED starts)
            current_player = obs["current_player"]
            current_agent = agent if current_player == 0 else opponent  # 0 for RED, 1 for YELLOW
            
            # Choose an action
            action = agent.choose_action(env)
            if isinstance(action, tuple):
                action = action[0]  # Ensure it's an integer

            
            # Execute the action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Accumulate the reward (from the perspective of the evaluated agent)
            if current_player == 0:  # If our agent played
                game_reward += reward
        
        # Analyze the result
        if "winner" in info:
            if info["winner"] == "RED":  # Evaluated agent won
                wins += 1
            elif info["winner"] == "YELLOW":  # Opponent won
                losses += 1
            else:  # Draw
                draws += 1
        else:
            # If there's no winner info, count based on the board state
            red_count, yellow_count = env._get_score()
            if red_count > yellow_count:
                wins += 1
            elif yellow_count > red_count:
                losses += 1
            else:
                draws += 1
        
        total_rewards += game_reward
    
    # Calculate statistics
    win_rate = wins / num_games
    avg_reward = total_rewards / num_games
    
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "avg_reward": avg_reward
    }
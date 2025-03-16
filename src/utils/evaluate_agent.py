from tqdm import tqdm
from Env.env import OthelloEnv


def evaluate_agent(agent, opponent, num_games=100, render=False):
    """Evaluates an agent against an opponent over multiple games."""
    env = OthelloEnv()
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
            # Determine which agent plays (BLACK starts)
            current_player = obs["current_player"]
            current_agent = agent if current_player == 0 else opponent  # 0 for BLACK, 1 for WHITE
            
            # Choose an action
            action = current_agent.choose_action(env)
            
            # Execute the action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Accumulate the reward (from the perspective of the evaluated agent)
            if current_player == 0:  # If our agent played
                game_reward += reward
        
        # Analyze the result
        black_count, white_count = env._get_score()
        if black_count > white_count:  # BLACK won
            wins += 1
        elif white_count > black_count:  # WHITE won
            losses += 1
        else:  # Draw
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
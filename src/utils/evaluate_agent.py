from tqdm import tqdm
from Env.env import OthelloEnv


def evaluate_agent(agent, opponent, num_games=100, render=False):
    """Évalue un agent contre un adversaire sur plusieurs parties."""
    env = OthelloEnv()
    wins = 0
    losses = 0
    draws = 0
    total_rewards = 0
    
    for game in tqdm(range(num_games), desc=f"{agent.name} vs {opponent.name}"):
        # Réinitialiser l'environnement
        obs, _ = env.reset()
        done = False
        game_reward = 0
        
        # Jouer la partie
        while not done:
            # Déterminer quel agent joue (BLACK commence)
            current_player = obs["current_player"]
            current_agent = agent if current_player == 0 else opponent  # 0 pour BLACK, 1 pour WHITE
            
            # Choisir une action
            action = current_agent.choose_action(env)
            
            # Exécuter l'action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Accumuler la récompense (du point de vue de l'agent évalué)
            if current_player == 0:  # Si c'est notre agent qui a joué
                game_reward += reward
        
        # Analyser le résultat
        black_count, white_count = env._get_score()
        if black_count > white_count:  # BLACK a gagné
            wins += 1
        elif white_count > black_count:  # WHITE a gagné
            losses += 1
        else:  # Match nul
            draws += 1
        
        total_rewards += game_reward
    
    # Calculer les statistiques
    win_rate = wins / num_games
    avg_reward = total_rewards / num_games
    
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "avg_reward": avg_reward
    }
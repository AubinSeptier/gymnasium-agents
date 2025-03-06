# main.py
from environment import Environment
from mcts_node import MonteCarloTreeSearchNode
from state import OthelloState
import curses
import time

KEY_MAPPING = {
    curses.KEY_UP: 2,       # Flèche haut → UP
    curses.KEY_DOWN: 5,     # Flèche bas → DOWN
    curses.KEY_LEFT: 4,     # Flèche gauche → LEFT
    curses.KEY_RIGHT: 3,    # Flèche droite → RIGHT
    ord(' '): 1,            # Espace → FIRE
    ord('b'): -1,           # 'b' → Forcer le changement de joueur (debug)
}

def othello_state_from_observation(observation):

    current_player = 1
    return OthelloState(board=observation, current_player=current_player)

def game_loop(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(1)
    stdscr.timeout(250)

    # Initialize environment with human render mode.
    env = Environment(env_name="ALE/Othello-v5", render_mode="human")
    observation, info = env.reset()

    is_bot_turn = False   # Human starts.
    bot_delay = 1.5       # Delay (in seconds) before bot moves.
    last_action_time = 0
    successful_move = False
    action_space_info = str(env.action_space)

    while True:
        current_time = time.time()
        stdscr.erase()
        stdscr.clear()

        stdscr.addstr(0, 0, "🎮 Contrôles : Flèches = Mouvement | ESPACE = Fire | Q = Quitter | B = Forcer changement")
        stdscr.addstr(1, 0, f"🎯 Espace d'action : {action_space_info}")
        stdscr.addstr(2, 0, f"👥 Tour actuel : {'>>> 🤖 BOT <<<' if is_bot_turn else '>>> 👤 JOUEUR <<<'}", curses.A_BOLD)

        action = None
        is_fire_action = False

        key = stdscr.getch()
        if key == ord('q'):
            break
        if key == ord('b'):
            is_bot_turn = not is_bot_turn
            last_action_time = current_time
            stdscr.addstr(10, 0, "⚠️ Changement forcé de joueur", curses.A_BOLD)
            stdscr.refresh()
            time.sleep(1)

        # Human turn: use key mapping.
        if not is_bot_turn:
            if key in KEY_MAPPING and KEY_MAPPING[key] >= 0:
                action = KEY_MAPPING[key]
                is_fire_action = (action == 1)
        # Bot turn: use MCTS after a delay.
        elif is_bot_turn and current_time - last_action_time >= bot_delay:
            # Convert current observation to OthelloState.
            current_state = othello_state_from_observation(observation)
            mcts_root = MonteCarloTreeSearchNode(state=current_state)
            bot_move = mcts_root.best_action(simulations_number=100)
            stdscr.addstr(10, 0, f" BOT a choisi: {bot_move}", curses.A_BOLD)
            action = 1
            is_fire_action = True
            last_action_time = current_time

        if action is not None:
            observation, reward, terminated, truncated, info = env.step(action)
            stdscr.addstr(4, 0, f" Action jouée : {action} ({'Bot' if is_bot_turn else 'Joueur'})")
            stdscr.addstr(5, 0, f" Récompense : {reward}")
            stdscr.addstr(6, 0, f" État du jeu : {'Terminé' if terminated else 'En cours'}")
            stdscr.addstr(7, 0, f"ℹ️ Action FIRE: {' Oui' if is_fire_action else '❌ Non'}")

            if is_fire_action:
                successful_move = (reward != 0)
                stdscr.addstr(8, 0, f"ℹ️ Mouvement réussi (Reward != 0): {' Oui' if successful_move else '❌ Non'}")
                if successful_move:
                    is_bot_turn = not is_bot_turn
                    last_action_time = current_time
            else:
                stdscr.addstr(8, 0, "ℹ️ Mouvement réussi (Action Direction): N/A")
            stdscr.addstr(9, 0, f"👥 Prochain tour : {' BOT' if is_bot_turn else '👤 JOUEUR'}")

            if terminated or truncated:
                stdscr.addstr(10, 0, " Partie terminée ! Réinitialisation...", curses.A_BOLD)
                stdscr.refresh()
                time.sleep(2)
                observation, info = env.reset()
                is_bot_turn = False

        if is_bot_turn:
            time_to_next = max(0, bot_delay - (current_time - last_action_time))
            stdscr.addstr(11, 0, f" Bot joue dans: {time_to_next:.1f}s")

        stdscr.refresh()

    env.close()

if __name__ == "__main__":
    import curses
    curses.wrapper(game_loop)

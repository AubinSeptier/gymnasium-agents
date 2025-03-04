from environment import Environment
import curses
import random
import time

KEY_MAPPING = {
    curses.KEY_UP: 2,       # Flèche haut → UP
    curses.KEY_DOWN: 5,     # Flèche bas → DOWN
    curses.KEY_LEFT: 4,     # Flèche gauche → LEFT
    curses.KEY_RIGHT: 3,    # Flèche droite → RIGHT
    ord(' '): 1,           # Espace → FIRE
    ord('b'): -1,          # 'b' → Forcer le changement de joueur (debug)
}

def game_loop(stdscr):
    """ Boucle principale du jeu en utilisant `curses` """
    curses.curs_set(0)  # Cache le curseur
    stdscr.nodelay(1)   # Ne pas bloquer en attente d'une touche
    stdscr.timeout(250) # Augmentation du timeout à 250ms

    # Initialisation de l'environnement
    env = Environment(env_name="ALE/Othello-v5", render_mode="human")
    observation, info = env.reset()

    # Variables d'état
    is_bot_turn = False    # Le joueur humain commence
    bot_delay = 1.5        # Délai avant que le bot joue (en secondes)
    last_action_time = 0   # Temps de la dernière action
    successful_move = False

    # Afficher les informations sur l'espace d'action
    action_space_info = str(env.action_space)

    while True:
        current_time = time.time()
        stdscr.erase()  # Remplir l'écran avec des espaces avant d'effacer
        stdscr.clear()

        # Affichage des contrôles et de l'état du jeu
        stdscr.addstr(0, 0, "🎮 Contrôles : Flèches = Mouvement | ESPACE = Fire | Q = Quitter | B = Forcer changement")
        stdscr.addstr(1, 0, f"🎯 Espace d'action : {action_space_info}")
        stdscr.addstr(2, 0, f"👥 Tour actuel : {'>>> 🤖 BOT <<<' if is_bot_turn else '>>> 👤 JOUEUR <<<'}", curses.A_BOLD) # Indicateurs visuels FORTS

        # Gestion de l'action de jeu
        action = None
        is_fire_action = False # Indique si l'action est FIRE

        # Lecture de la touche pressée (pour le joueur humain ou pour quitter)
        key = stdscr.getch()

        if key == ord('q'):  # Touche "q" pour quitter
            break

        if key == ord('b'):  # Touche "b" pour forcer le changement de joueur (debug)
            is_bot_turn = not is_bot_turn
            last_action_time = current_time
            stdscr.addstr(10, 0, "⚠️ Changement forcé de joueur", curses.A_BOLD)
            stdscr.refresh()
            time.sleep(1)  # Pause pour afficher le message

        # Tour du joueur humain
        if not is_bot_turn:
            if key in KEY_MAPPING and KEY_MAPPING[key] >= 0:
                action = KEY_MAPPING[key]
                if action == 1: # Action FIRE
                    is_fire_action = True
                else:
                    is_fire_action = False # Action de mouvement

        # Tour du bot (avec délai)
        elif is_bot_turn and current_time - last_action_time >= bot_delay:
            possible_actions = [value for key, value in KEY_MAPPING.items() if value >= 0] # Recalculer possible_actions (bonne pratique)
            print(f"🤖 Tour du BOT - Actions possibles: {possible_actions}") # Affiche les actions possibles pour le debug
            action = 1  # Action FIRE FIXE pour le bot (déterministe) - Le bot essaie de placer un pion
            is_fire_action = True # Le bot fait l'action FIRE
            print(f"🤖 BOT choisit l'action FIXE: {action} (FIRE)") # Affiche l'action choisie par le bot pour le debug
            last_action_time = current_time

        # Exécution de l'action si définie
        if action is not None:
            print(f"🎲 Action jouée : {action} ({'Bot' if is_bot_turn else 'Joueur'})") # Affiche l'action jouée dans le terminal
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"🏆 Récompense : {reward}") # Affiche la récompense dans le terminal pour le debug
            print(f"ℹ️ Info: {info}") # Affiche le dictionnaire info dans le terminal pour le debug

            stdscr.addstr(4, 0, f"🎲 Action jouée : {action} ({'Bot' if is_bot_turn else 'Joueur'})")
            stdscr.addstr(5, 0, f"🏆 Récompense : {reward}")
            stdscr.addstr(6, 0, f"📊 État du jeu : {'Terminé' if terminated else 'En cours'}")
            stdscr.addstr(7, 0, f"ℹ️ Action FIRE: {'✅ Oui' if is_fire_action else '❌ Non'}") # Indique si c'est FIRE

            if is_fire_action: # Changer de tour seulement si action FIRE
                successful_move = reward != 0  # successful_move basé sur la récompense SI action FIRE
                stdscr.addstr(8, 0, f"ℹ️ Mouvement réussi (Reward != 0): {'✅ Oui' if successful_move else '❌ Non'}")
                if successful_move:
                    is_bot_turn = not is_bot_turn
                    last_action_time = current_time
            else:
                 stdscr.addstr(8, 0, f"ℹ️ Mouvement réussi (Action Direction): N/A") # Non applicable pour action direction

            stdscr.addstr(9, 0, f"👥 Prochain tour : {'🤖 BOT' if is_bot_turn else '👤 JOUEUR'}") # Ligne décalée


            # Réinitialisation si le jeu est terminé
            if terminated or truncated:
                stdscr.addstr(10, 0, "💀 Partie terminée ! Réinitialisation...", curses.A_BOLD)
                stdscr.refresh()
                time.sleep(2)  # Pause pour voir le message
                observation, info = env.reset()
                is_bot_turn = False  # Le joueur humain recommence

        # Afficher le temps restant avant l'action du bot
        if is_bot_turn:
            time_to_next = max(0, bot_delay - (current_time - last_action_time))
            stdscr.addstr(11, 0, f"⏱️ Bot joue dans: {time_to_next:.1f}s") # Ligne décalée

        stdscr.refresh()  # Rafraîchir l'affichage

    env.close()

# Lancer `curses.wrapper` pour gérer proprement l'affichage en mode terminal
curses.wrapper(game_loop)
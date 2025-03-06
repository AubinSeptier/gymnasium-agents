# main_botvsbot.py
import curses
import time
from environment import Environment
from mcts_node import MonteCarloTreeSearchNode
from state import OthelloState


ACTION_FIRE   = 1
ACTION_UP     = 2
ACTION_RIGHT  = 3
ACTION_LEFT   = 4
ACTION_DOWN   = 5

def map_move_to_actions(cursor_r, cursor_c, target_r, target_c):
    """
    Return a list of actions that moves the cursor from (cursor_r, cursor_c)
    to (target_r, target_c) and then presses FIRE.
    Adjust if your environment has different action codes.
    """
    actions = []
    
    while cursor_r < target_r:
        actions.append(ACTION_DOWN)
        cursor_r += 1
    while cursor_r > target_r:
        actions.append(ACTION_UP)
        cursor_r -= 1
   
    while cursor_c < target_c:
        actions.append(ACTION_RIGHT)
        cursor_c += 1
    while cursor_c > target_c:
        actions.append(ACTION_LEFT)
        cursor_c -= 1
    
    actions.append(ACTION_FIRE)
    return actions

def othello_state_from_observation(observation):
    return OthelloState()

def game_loop(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(1)
    stdscr.timeout(250)

    
    env = Environment(env_name="ALE/Othello-v5", render_mode="human")
    observation, info = env.reset()

    bot1_cursor = (0, 0)
    bot2_cursor = (7, 7)


    current_player = 1

    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "Bot vs Bot in ALE Othello\n")
        stdscr.addstr(1, 0, f"Current player: {'Bot1' if current_player == 1 else 'Bot2'}")

        
        current_state = othello_state_from_observation(observation)
        mcts_root = MonteCarloTreeSearchNode(state=current_state)
        best_move = mcts_root.best_action(simulations_number=50)

        if best_move is None:
            
            stdscr.addstr(2, 0, "No moves found (pass).")
            
            observation, reward, terminated, truncated, info = env.step(ACTION_FIRE)
            time.sleep(1)
        else:
            
            if current_player == 1:
                cursor_r, cursor_c = bot1_cursor
            else:
                cursor_r, cursor_c = bot2_cursor

            target_r, target_c = best_move
            actions = map_move_to_actions(cursor_r, cursor_c, target_r, target_c)

           
            for act in actions:
                observation, reward, terminated, truncated, info = env.step(act)
                time.sleep(0.3) 
                if terminated or truncated:
                    break

            
            if current_player == 1:
                bot1_cursor = (target_r, target_c)
            else:
                bot2_cursor = (target_r, target_c)

        
        if terminated or truncated:
            stdscr.addstr(3, 0, "Game over!")
            stdscr.refresh()
            time.sleep(2)
            break

        
        current_player = -current_player

        stdscr.refresh()

    env.close()

def main():
    curses.wrapper(game_loop)

if __name__ == "__main__":
    main()

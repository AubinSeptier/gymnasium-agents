import numpy as np
import random
import math
import time
import copy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Any, Union
from Env.env import OthelloEnv, BLACK, WHITE, EMPTY, BOARD_SIZE


class MCTSNode:
    """A node in the MCTS tree, optimized for parallelization."""
    
    __slots__ = ('env', 'parent', 'action', 'children', 'visits', 'reward', 
                'untried_actions', 'current_player', 'terminal', 'fully_expanded')
    
    def __init__(self, env: OthelloEnv, parent=None, action=None):
        self.env = env  # Environment state
        self.parent = parent  # Parent node
        self.action = action  # Action that led to this node
        self.children = []  # Child nodes
        self.visits = 0  # Number of visits
        self.reward = 0.0  # Cumulative reward
        self.untried_actions = None  # Unexplored actions (calculated on demand)
        self.current_player = 1 if env.current_player == BLACK else -1  # Current player
        self.terminal = None  # Terminal state indicator (calculated on demand)
        self.fully_expanded = False  # Fully expanded indicator
    
    def _get_untried_actions(self) -> List[int]:
        """Returns all legal actions not yet tried from this state."""
        if self.untried_actions is None:
            valid_moves_array = self.env._get_observation()["valid_moves"]
            self.untried_actions = [i for i, is_valid in enumerate(valid_moves_array) if is_valid == 1]
        return self.untried_actions.copy()  # Copy to avoid external modifications
    
    def is_fully_expanded(self) -> bool:
        """Checks if all possible moves from this node have been explored."""
        if not self.fully_expanded:
            self.fully_expanded = len(self._get_untried_actions()) == 0
        return self.fully_expanded
    
    def is_terminal(self) -> bool:
        """Checks if this state is terminal (end of game)."""
        if self.terminal is None:
            obs = self.env._get_observation()
            self.terminal = np.sum(obs["valid_moves"]) == 0  # No valid moves
        return self.terminal
    
    def get_reward(self) -> float:
        """Returns the reward for this terminal state."""
        black_count, white_count = self.env._get_score()
        if black_count > white_count:
            return 1.0 if self.current_player == 1 else -1.0
        elif white_count > black_count:
            return -1.0 if self.current_player == 1 else 1.0
        else:
            return 0.0  # Draw
    
    def best_child(self, c_param: float = 1.4) -> 'MCTSNode':
        """Selects the best child according to the UCB formula."""
        # Optimization: preallocate the array for weights
        choices_weights = np.zeros(len(self.children))
        
        # Vectorize UCB calculation for efficiency
        for i, child in enumerate(self.children):
            # UCB formula: exploitation (Q/N) + exploration (c * sqrt(ln(N_parent) / N_child))
            exploitation = child.reward / child.visits
            exploration = c_param * math.sqrt((2 * math.log(self.visits)) / child.visits)
            choices_weights[i] = exploitation + exploration
        
        return self.children[np.argmax(choices_weights)]
    
    def expand(self) -> 'MCTSNode':
        """Adds a new child to the node by randomly choosing an unexplored action."""
        untried_actions = self._get_untried_actions()
        if not untried_actions:
            return self  # No actions to explore
            
        action = untried_actions.pop()
        self.untried_actions = untried_actions  # Update unexplored actions
        
        # Copy the environment and apply the action
        new_env = copy.deepcopy(self.env)
        _, _, _, _, _ = new_env.step(action)
        
        # Create a new node
        child_node = MCTSNode(new_env, parent=self, action=action)
        self.children.append(child_node)
        
        # Check if all actions have been explored
        if not self.untried_actions:
            self.fully_expanded = True
            
        return child_node
    
    def simulate(self) -> float:
        """Simulates a game from this state to a terminal state by choosing random actions."""
        # Local copy of the environment for simulation
        sim_env = copy.deepcopy(self.env)
        terminated = False
        
        # Limit the number of steps to avoid infinite loops
        max_steps = 60  # Arbitrary limit
        step_count = 0
        
        while not terminated and step_count < max_steps:
            # Get valid moves efficiently
            obs = sim_env._get_observation()
            valid_moves_indices = np.where(obs["valid_moves"] == 1)[0]
            
            # If no valid moves, the game is over
            if len(valid_moves_indices) == 0:
                break
            
            # Choose a random action (faster with numpy)
            action = np.random.choice(valid_moves_indices)
            
            # Execute the action
            _, _, terminated, _, _ = sim_env.step(action)
            step_count += 1
        
        # Calculate final reward
        black_count, white_count = sim_env._get_score()
        if black_count > white_count:
            return 1.0 if self.current_player == 1 else -1.0
        elif white_count > black_count:
            return -1.0 if self.current_player == 1 else 1.0
        else:
            return 0.0  # Draw
    
    def backpropagate(self, result: float) -> None:
        """Updates the statistics of this node and its parents with the result of the simulation."""
        # Use an iterative loop instead of recursion to avoid stack errors
        node = self
        while node is not None:
            node.visits += 1
            node.reward += result
            node = node.parent


def run_simulation(root_state, exploration_weight=1.4):
    """Function to run a complete MCTS simulation (selection, expansion, simulation, backpropagation)."""
    # Create a copy of the root state
    env_copy = copy.deepcopy(root_state)
    node = MCTSNode(env_copy)
    
    # Selection
    while not node.is_terminal() and node.is_fully_expanded():
        node = node.best_child(exploration_weight)
    
    # Expansion
    if not node.is_terminal() and not node.is_fully_expanded():
        node = node.expand()
    
    # Simulation
    result = node.simulate()
    
    # Backpropagation
    node.backpropagate(result)
    
    # Return result and action
    return result, node.action


# Class of aggregated results for parallelism
class SimulationResults:
    """Class to aggregate the results of parallel simulations."""
    
    def __init__(self):
        self.action_visits = {}  # {action: number of visits}
        self.action_rewards = {}  # {action: total reward}
    
    def add_result(self, action, reward):
        """Adds a simulation result."""
        if action in self.action_visits:
            self.action_visits[action] += 1
            self.action_rewards[action] += reward
        else:
            self.action_visits[action] = 1
            self.action_rewards[action] = reward
    
    def merge(self, other):
        """Merges with another SimulationResults object."""
        for action, visits in other.action_visits.items():
            if action in self.action_visits:
                self.action_visits[action] += visits
                self.action_rewards[action] += other.action_rewards[action]
            else:
                self.action_visits[action] = visits
                self.action_rewards[action] = other.action_rewards[action]
    
    def best_action(self):
        """Returns the most visited action."""
        if not self.action_visits:
            return None
        return max(self.action_visits.items(), key=lambda x: x[1])[0]


def run_batch_simulations(env, batch_size, exploration_weight=1.4):
    """Runs a batch of simulations and returns the aggregated results."""
    results = SimulationResults()
    
    for _ in range(batch_size):
        reward, action = run_simulation(env, exploration_weight)
        results.add_result(action, reward)
    
    return results


class MCTSAgent:
    """MCTS Agent with parallelization for improved performance."""
    
    def __init__(self, num_simulations: int = 500, exploration_weight: float = 1.4, 
                 num_processes: int = None, batch_size: int = None):
        self.num_simulations = num_simulations  # Total number of simulations
        self.exploration_weight = exploration_weight  # Exploration weight
        
        # Determine the number of processes to use
        if num_processes is None:
            self.num_processes = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
        else:
            self.num_processes = num_processes
        
        # Determine the batch size per process
        if batch_size is None:
            # Divide the simulations evenly among the processes
            self.batch_size = max(1, self.num_simulations // self.num_processes)
        else:
            self.batch_size = batch_size
        
        # Adjust the total number of simulations to be divisible
        self.actual_num_simulations = self.batch_size * self.num_processes
        
        self.name = f"MCTS_{self.actual_num_simulations}"
        self.pool = None  # ProcessPoolExecutor will be initialized on demand
    
    def _initialize_pool(self):
        """Initializes the process pool if needed."""
        if self.pool is None:
            self.pool = ProcessPoolExecutor(max_workers=self.num_processes)
    
    def choose_action(self, env: OthelloEnv) -> int:
        """Chooses the best action according to parallelized MCTS."""
        # Check if valid actions exist
        obs = env._get_observation()
        valid_moves = [i for i, is_valid in enumerate(obs["valid_moves"]) if is_valid == 1]
        
        # If no valid moves, return a default action
        if not valid_moves:
            return 0
        
        # If only one valid move, return it immediately
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Initialize the process pool if needed
        self._initialize_pool()
        
        # Run simulations in parallel
        results = SimulationResults()
        batch_futures = []
        
        try:
            # Submit tasks
            for _ in range(self.num_processes):
                future = self.pool.submit(run_batch_simulations, env, self.batch_size, self.exploration_weight)
                batch_futures.append(future)
            
            # Retrieve results
            for future in as_completed(batch_futures):
                batch_results = future.result()
                results.merge(batch_results)
        
        except Exception as e:
            print(f"Exception during parallel MCTS calculation: {e}")
            # Fallback to a random valid action in case of error
            return random.choice(valid_moves)
        
        # Return the most visited action
        best_action = results.best_action()
        return best_action if best_action is not None else random.choice(valid_moves)
    
    def __del__(self):
        """Destructor to clean up resources."""
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None


# class MCTSAgent:
#     """MCTS Agent with parallelization for improved performance."""
    
#     def __init__(self, num_simulations: int = 500, exploration_weight: float = 1.4, 
#                  num_processes: int = None, batch_size: int = None, time_limit: float = None):
#         self.num_simulations = num_simulations  # Total number of simulations
#         self.exploration_weight = exploration_weight  # Exploration weight
#         self.time_limit = time_limit  # Time limit in seconds (if specified)
        
#         # Determine the number of processes to use
#         if num_processes is None:
#             self.num_processes = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
#         else:
#             self.num_processes = num_processes
        
#         # Determine the batch size per process
#         if batch_size is None:
#             # For time-based search, use an adaptive batch size strategy
#             if time_limit is not None:
#                 # Start with a moderate batch size - will be adjusted based on game state
#                 self.batch_size = 20
#             else:
#                 # Divide the simulations evenly among the processes
#                 self.batch_size = max(1, self.num_simulations // self.num_processes)
#         else:
#             self.batch_size = batch_size
        
#         # Set name based on mode
#         if time_limit is None:
#             self.actual_num_simulations = self.num_simulations
#             self.name = f"MCTS_{self.actual_num_simulations}"
#         else:
#             self.actual_num_simulations = 0  # Will be determined during search
#             self.name = f"MCTS_{time_limit}s"
        
#         self.pool = None  # ProcessPoolExecutor will be initialized on demand
        
#         # Performance tracking for adaptive behavior
#         self.move_count = 0
#         self.avg_simulation_time = 0.01  # Starting estimate (will be updated)
    
#     def _initialize_pool(self):
#         """Initializes the process pool if needed."""
#         if self.pool is None:
#             self.pool = ProcessPoolExecutor(max_workers=self.num_processes)
    
#     def choose_action(self, env: OthelloEnv) -> int:
#         """Chooses the best action according to parallelized MCTS."""
#         # Check if valid actions exist
#         obs = env._get_observation()
#         valid_moves = [i for i, is_valid in enumerate(obs["valid_moves"]) if is_valid == 1]
        
#         # If no valid moves, return a default action
#         if not valid_moves:
#             return 0
        
#         # If only one valid move, return it immediately
#         if len(valid_moves) == 1:
#             return valid_moves[0]
        
#         # Track move number for adaptive behavior
#         self.move_count += 1
        
#         # Initialize the process pool if needed
#         self._initialize_pool()
        
#         # Run simulations in parallel
#         results = SimulationResults()
        
#         # Start timing if using time-based search
#         start_time = time.time()
#         total_simulations = 0
        
#         try:
#             if self.time_limit is not None:
#                 # Estimate game stage based on valid moves count and move count
#                 valid_move_count = len(valid_moves)
#                 game_progress = min(1.0, self.move_count / 60.0)  # Assuming ~60 moves per game
                
#                 # Adjust batch size based on game stage
#                 # Early game: smaller batches (game tree is deeper)
#                 # Late game: larger batches (game tree is shallower)
#                 adjusted_batch_size = int(self.batch_size * (1 + 2 * game_progress))
                
#                 # For very late game (few valid moves), use larger batches
#                 if valid_move_count < 4:
#                     adjusted_batch_size *= 2
                
#                 # Early game strategy - immediate parallel batches with safety margin
#                 if game_progress < 0.3:  # First 30% of the game
#                     # Calculate estimated simulations based on previous performance
#                     estimated_simulations = int(0.9 * self.time_limit / self.avg_simulation_time)
                    
#                     # Cap at a reasonable level for parallelization
#                     max_initial_simulations = 100 if game_progress < 0.1 else 150
#                     target_simulations = min(estimated_simulations, max_initial_simulations)
                    
#                     # Submit parallel batches
#                     remaining_simulations = target_simulations
#                     batch_futures = []
                    
#                     while remaining_simulations > 0:
#                         # Determine batch size for this process
#                         current_batch_size = min(adjusted_batch_size, remaining_simulations)
                        
#                         # Submit the task
#                         future = self.pool.submit(run_batch_simulations, env, current_batch_size, self.exploration_weight)
#                         batch_futures.append(future)
                        
#                         remaining_simulations -= current_batch_size
                        
#                         # Limit parallelism to avoid overloading
#                         if len(batch_futures) >= self.num_processes:
#                             break
                    
#                     # Process results and calculate time for a simulation
#                     sim_start_time = time.time()
#                     sim_count = 0
                    
#                     for future in as_completed(batch_futures):
#                         batch_results = future.result()
#                         results.merge(batch_results)
#                         sim_count += current_batch_size  # This is approximate
#                         total_simulations += current_batch_size
                        
#                         # Check if we're approaching the time limit
#                         if time.time() - start_time > 0.8 * self.time_limit:
#                             # Cancel remaining futures if we're short on time
#                             for f in batch_futures:
#                                 if not f.done():
#                                     f.cancel()
#                             break
                    
#                     # Update average simulation time for future reference
#                     sim_elapsed = time.time() - sim_start_time
#                     if sim_count > 0 and sim_elapsed > 0:
#                         # Exponential moving average to smooth out variations
#                         self.avg_simulation_time = 0.7 * self.avg_simulation_time + 0.3 * (sim_elapsed / sim_count)
                    
#                     # If we have time left, use the full time
#                     remaining_time = self.time_limit - (time.time() - start_time)
#                     if remaining_time > 0.2:  # At least 200ms left
#                         # Calculate how many more we can run
#                         additional_simulations = int(0.8 * remaining_time / self.avg_simulation_time)
                        
#                         if additional_simulations > 0:
#                             # Run these in parallel batches
#                             batch_futures = []
#                             remaining_simulations = additional_simulations
                            
#                             while remaining_simulations > 0:
#                                 # Determine batch size for this process
#                                 current_batch_size = min(adjusted_batch_size, remaining_simulations)
                                
#                                 # Submit the task
#                                 future = self.pool.submit(run_batch_simulations, env, current_batch_size, self.exploration_weight)
#                                 batch_futures.append(future)
                                
#                                 remaining_simulations -= current_batch_size
                                
#                                 # Limit parallelism to avoid overloading
#                                 if len(batch_futures) >= self.num_processes:
#                                     break
                            
#                             # Process results
#                             for future in as_completed(batch_futures):
#                                 if time.time() - start_time >= 0.95 * self.time_limit:
#                                     # Cancel remaining futures if we're very close to time limit
#                                     for f in batch_futures:
#                                         if not f.done():
#                                             f.cancel()
#                                     break
                                    
#                                 batch_results = future.result()
#                                 results.merge(batch_results)
#                                 total_simulations += current_batch_size
                
#                 # Mid to late game strategy - more aggressive parallelization
#                 else:
#                     # Calculate how many simulations we can safely run based on game stage
#                     estimated_simulations = int(0.85 * self.time_limit / self.avg_simulation_time)
                    
#                     # Determine target simulations - more aggressive in late game
#                     target_factor = 1.0 + min(1.0, (game_progress - 0.3) * 2)  # 1.0 to 2.0 scaling factor
#                     target_simulations = int(estimated_simulations * target_factor)
                    
#                     # Submit parallel batches
#                     batch_futures = []
#                     remaining_simulations = target_simulations
                    
#                     while remaining_simulations > 0:
#                         # Determine batch size for this process
#                         current_batch_size = min(adjusted_batch_size, remaining_simulations)
                        
#                         # Submit the task
#                         future = self.pool.submit(run_batch_simulations, env, current_batch_size, self.exploration_weight)
#                         batch_futures.append(future)
                        
#                         remaining_simulations -= current_batch_size
                        
#                         # Limit parallelism to available processors
#                         if len(batch_futures) >= self.num_processes:
#                             break
                    
#                     # Process results
#                     for future in as_completed(batch_futures):
#                         if time.time() - start_time >= 0.95 * self.time_limit:
#                             # Cancel remaining futures if we're very close to time limit
#                             for f in batch_futures:
#                                 if not f.done():
#                                     f.cancel()
#                             break
                            
#                         batch_results = future.result()
#                         results.merge(batch_results)
#                         total_simulations += current_batch_size
                        
#                         # Update simulation time estimate from this batch
#                         sim_elapsed = time.time() - start_time
#                         if total_simulations > 0 and sim_elapsed > 0:
#                             self.avg_simulation_time = sim_elapsed / total_simulations
                
#                 # Update actual number of simulations for reference
#                 self.actual_num_simulations = total_simulations
                
#             else:
#                 # Original simulation-based approach
#                 remaining_simulations = self.num_simulations
                
#                 # Submit tasks in batches that make sense for the total simulation count
#                 while remaining_simulations > 0:
#                     batch_futures = []
                    
#                     # Calculate batch size for this round
#                     current_batch_size = min(self.batch_size, remaining_simulations)
#                     batches_to_run = min(self.num_processes, remaining_simulations // current_batch_size)
                    
#                     # Submit the batch of tasks
#                     for _ in range(batches_to_run):
#                         future = self.pool.submit(run_batch_simulations, env, current_batch_size, self.exploration_weight)
#                         batch_futures.append(future)
                    
#                     # Retrieve results
#                     for future in as_completed(batch_futures):
#                         batch_results = future.result()
#                         results.merge(batch_results)
                    
#                     # Update remaining simulations
#                     remaining_simulations -= (current_batch_size * batches_to_run)
        
#         except Exception as e:
#             print(f"Exception during parallel MCTS calculation: {e}")
#             # Fallback to a random valid action in case of error
#             return random.choice(valid_moves)
        
#         # Print statistics
#         elapsed_time = time.time() - start_time
#         print(f"MCTS completed {self.actual_num_simulations} simulations in {elapsed_time:.2f} seconds")
        
#         # Return the most visited action
#         best_action = results.best_action()
#         return best_action if best_action is not None else random.choice(valid_moves)
    
#     def __del__(self):
#         """Destructor to clean up resources."""
#         if self.pool is not None:
#             self.pool.shutdown()
#             self.pool = None
import gymnasium as gym
import ale_py

class Environment(gym.Env):
    """
    Class for manage the environment.
    
    Attributes:
        env (gym.Env): the gymnasium environment.
        observation_space (ObsType): the observation space of the environment.
        action_space (ActType): the action space of the environment.
    """
    
    def __init__(self, env_name: str, render_mode: str = None):
        gym.register_envs(ale_py)
        self.env = gym.make(env_name, render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def step(self, action):
        """
        Run one timestep of the environment's dynamics. 
        
        Args:
            action (ActType): an action to take in the environment.
            
        Returns:
            observation (ObsType): agent's observation of the current environment.
            reward (float) : amount of reward returned after previous action.
            terminated (bool): whether the episode has ended.
            truncated (bool): whether the episode has been truncated.
            info (dict): contains auxiliary information.
        """
        return self.env.step(action)
    
    def reset(self, seed=None, options=None):
        """
        Reset the state of the environment and returns an initial observation.
        
        Args:
            seed (int): the seed to pass to the environment's random number generator.
            options (dict): the options to pass to the environment's reset method.
            
        Returns:
            observation (ObsType): the initial observation of the space.
        """
        return self.env.reset(seed=seed, options=options)
    
    def close(self):
        """
        Close the environment and free resources.
        """
        self.env.close()
        
    def get_space(self):
        """
        Get the observation and action space of the environment.
        
        Returns:
            observation_space (ObsType): the observation space of the environment.
            action_space (ActType): the action space of the environment.
        """
        return self.observation_space, self.action_space
        
        
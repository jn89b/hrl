import gymnasium
import ray
import numpy as np
import os
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from ray.rllib.policy.policy import PolicySpec
#import algorithms
from stable_baselines3 import PPO as StablePPO
from ray.rllib.algorithms import Algorithm
from config import PWD_DEFENSIVE, PWD_OFFENSIVE  # Assuming these are defined in config.py

# https://github.com/ray-project/ray/blob/master/rllib/examples/envs/classes/windy_maze_env.py

USE_STABLE = True

def load_trained_agents() -> tuple:
    """
    Super hacky but this is the only way to get this to work for now
    """
    trained_defensive = "PPO_DefensivePolicyEnv/trained_0"
    defense_path = os.path.join(PWD_DEFENSIVE, trained_defensive+"/checkpoint_000001")

    trained_offensive = "PPO_OffensivePolicyEnv/trained_0"
    offense_path = os.path.join(PWD_OFFENSIVE, trained_offensive+"/checkpoint_000001")

    # Load the configurations for the trained policies
    config_offensive = PPOConfig().framework("torch").environment(env=OffensivePolicyEnv).to_dict()
    config_defensive = PPOConfig().framework("torch").environment(env=DefensivePolicyEnv).to_dict()

    # Load the offensive policy from the checkpoint
    offensive_config = PPOConfig().framework("torch").environment(env=OffensivePolicyEnv)
    offensive_agent = offensive_config.build().from_checkpoint(offense_path)

    # Load the defensive policy from the checkpoint
    defensive_config = PPOConfig().framework("torch").environment(env=DefensivePolicyEnv)
    defensive_agent = defensive_config.build().from_checkpoint(defense_path)
    return offensive_agent, defensive_agent

def load_stable_agents(offensive_env, defensive_env) -> tuple:
    defensive_name = "ppo_defensive"
    offensive_name = "ppo_offensive"

    offensive_env = OffensivePolicyEnv()
    offensive_agent = StablePPO.load(offensive_name, env=offensive_env)
    # offensive_agent.set_env(offensive_env)

    defensive_env = DefensivePolicyEnv()
    defensive_agent = StablePPO.load(defensive_name, env=defensive_env)
    return offensive_agent, defensive_agent

# Defensive Policy Environment
class DefensivePolicyEnv(gymnasium.Env):
    def __init__(self, env_config=None):
        super(DefensivePolicyEnv, self).__init__()
        self.grid_size = 5
        self.action_space = spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(4,), dtype=np.int32)
        self.reset()

    def reset(self, seed=None, options=None):
        self.agent_pos = np.random.randint(0, self.grid_size, size=2)
        self.enemy_pos = np.random.randint(0, self.grid_size, size=2)
        #make sure type 32
        return np.concatenate([self.agent_pos, self.enemy_pos]).astype(np.int32), {}
        # return np.concatenate([self.agent_pos, self.enemy_pos]), {}

    def step(self, action):
        # Move agent
        if action == 0:  # up
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1:  # right
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        elif action == 2:  # down
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)
        elif action == 3:  # left
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        
        # Move enemy randomly
        self.enemy_pos += np.random.choice([-1, 0, 1], size=2)
        self.enemy_pos = np.clip(self.enemy_pos, 0, self.grid_size - 1)
        
        obs = np.concatenate([self.agent_pos, self.enemy_pos])
        # reward = 1 if not np.array_equal(self.agent_pos, self.enemy_pos) else -10
        if np.array_equal(self.agent_pos, self.enemy_pos):
            # print("Collision!", self.agent_pos, self.enemy_pos)
            reward = -10
        else:
            reward = 1
        done = np.array_equal(self.agent_pos, self.enemy_pos)
        return obs, reward, done, False, {}

# Offensive Policy Environment
class OffensivePolicyEnv(gymnasium.Env):
    def __init__(self, env_config=None):
        super(OffensivePolicyEnv, self).__init__()
        self.grid_size = 5
        self.action_space = spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(4,), dtype=np.int32)
        # self.reset()

    def reset(self, seed=None, options=None):
        self.agent_pos = np.random.randint(0, self.grid_size, size=2)
        # self.target_pos = np.array([self.grid_size-1, self.grid_size-1])
        #randomize target position
        rand_x = np.random.randint(0, self.grid_size)
        rand_y = np.random.randint(0, self.grid_size)
        self.target_pos = np.array([rand_x, rand_y])
        #make sure int32
        self.agent_pos = self.agent_pos.astype(np.int32)
        self.target_pos = self.target_pos.astype(np.int32)
        return np.concatenate([self.agent_pos, self.target_pos]), {}

    def step(self, action):
        # Move agent
        if action == 0:  # up
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1:  # right
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        elif action == 2:  # down
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)
        elif action == 3:  # left
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        
        obs = np.concatenate([self.agent_pos, self.target_pos])
        #reward = 10 if np.array_equal(self.agent_pos, self.target_pos) else -1
        if np.array_equal(self.agent_pos, self.target_pos):
            reward = 10
            done = True
        else:
            reward = -1
        
        done = np.array_equal(self.agent_pos, self.target_pos)
        return obs, reward, done, False, {}

class HierarchicalEnv(gymnasium.Env):
    def __init__(self, env_config=None):
        self.defensive_env = DefensivePolicyEnv()
        self.offensive_env = OffensivePolicyEnv()
        self.current_env = self.defensive_env
        self.mode = "defensive"
        
        self.current_step = 0
        self.max_steps = 100 
        
        self.offensive_agent = None
        self.defensive_agent = None

        # Expand action space: 0 for defensive, 1 for offensive
        self.action_space = spaces.Discrete(2)  # Select between offensive and defensive
        self.policy_choice_history = []

        # Observation space includes agent position, target position, and enemy position
        # the grid size must match the environment
        self.observation_space = spaces.Box(
            low=0,
            high=self.defensive_env.grid_size - 1,
            shape=(6,),
            dtype=np.int32
        )
        
        if self.offensive_agent is None or self.defensive_agent is None:
            print("loading agents")
            if USE_STABLE:
                self.offensive_agent, self.defensive_agent = load_stable_agents(
                    offensive_env=self.offensive_env,
                    defensive_env=self.defensive_env
                )
            else:
                self.offensive_agent, self.defensive_agent = load_trained_agents()
        
    # def load_agents(self, offensive_agent, defensive_agent):
    #     self.offensive_agent = offensive_agent
    #     self.defensive_agent = defensive_agent

    def reset(self, seed=None, options=None):
        # Reset the current environment
        # obs = self.current_env.reset()
        #reset both environments
        self.policy_choice_history = []
        self.defensive_env.reset()
        self.offensive_env.reset()
        self.current_step = 0
        # Return combined observations
        # Combine agent's position with target and enemy positions
        # return {"agent_1": self._get_combined_obs()}, {}
        return self._get_combined_obs(), {}
    
    def get_defensive_obs(self):
        """
        The defensive observation only cares about its own position and the enemy's position.
        """
        return np.concatenate([self.defensive_env.agent_pos, self.defensive_env.enemy_pos])
    
    def get_offensive_obs(self):
        """
        The offensive observation cares about its own position and the target's position.
        """
        return np.concatenate([self.offensive_env.agent_pos, self.offensive_env.target_pos])
    
    
    def _get_combined_obs(self):
        #we are going to combine the information in the following order: ego agent, target, enemy
        ego_position = self.offensive_env.agent_pos 
        target_position = self.offensive_env.target_pos
        enemy_position = self.defensive_env.enemy_pos
        
        #make sure they are all float32
        return np.concatenate([ego_position.astype(np.int32), 
                               target_position.astype(np.int32), 
                               enemy_position.astype(np.int32)])
        
    
    def step(self, action_dict):
        # Check if agents are loaded before taking a step
        assert self.offensive_agent is not None, "Offensive agent is not loaded!"
        assert self.defensive_agent is not None, "Defensive agent is not loaded!"
        
        policy_choice = action_dict
        done = False
        if isinstance(policy_choice, tuple):
            policy_choice = policy_choice[0]  # Extract the first element if it's a tuple

        # Switch mode based on policy_choice
        if policy_choice == 0:
            self.switch_mode("defensive")
        else:
            self.switch_mode("offensive")

        #just to keep track of policy choices
        self.policy_choice_history.append(policy_choice)
        
        if self.mode == "defensive":
            if not USE_STABLE:
                action = self.defensive_agent.compute_single_action(
                    self.get_defensive_obs(), deterministic=True)
            else:
                action, _states = self.defensive_agent.predict(
                    self.get_defensive_obs(), deterministic=True)
        else:
            if not USE_STABLE:  
                action = self.offensive_agent.compute_single_action(self.get_offensive_obs())
            else:
                action, _states = self.offensive_agent.predict(self.get_offensive_obs())

        # Execute the action in the selected environment
        # obs, reward, done, info = self.current_env.step(action)\
        # do action to each environment and see what happens
        defensive_obs, defensive_reward, defensive_done, _, defensive_info = self.defensive_env.step(action)
        offensive_obs, offensive_reward, offensive_done, _, offensive_info = self.offensive_env.step(action)
        #print("done:", defensive_done, offensive_done)
        #sum the rewards
        #reward = defensive_reward + offensive_reward
        
        #get distance to target
        denemy = np.linalg.norm(self.offensive_env.agent_pos - self.offensive_env.target_pos)
        dgoal = np.linalg.norm(self.defensive_env.agent_pos - self.defensive_env.enemy_pos)
        
        reward = denemy - dgoal
        
        if self.current_step >= self.max_steps:
            done = True
            reward -= 10  # Penalty for exceeding max steps
            print("Max steps reached!")
        elif defensive_done or offensive_done:
            if offensive_done:
                reward += 10
            if defensive_done:
                reward -=100
            done = True
        
        self.current_step += 1
        
        obs = self._get_combined_obs()
        return obs, reward, done, _, {"defensive_info": defensive_info, "offensive_info": offensive_info} 

    def switch_mode(self, mode):
        if mode == "defensive":
            self.current_env = self.defensive_env
        else:
            self.current_env = self.offensive_env
        self.mode = mode



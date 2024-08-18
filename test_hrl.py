import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym import spaces
import numpy as np
from grid_env import OffensivePolicyEnv, DefensivePolicyEnv, HierarchicalEnv  # Assuming these are defined in grid_env.py
from config import PWD_DEFENSIVE, PWD_OFFENSIVE  # Assuming these are defined in config.py
# Initialize Ray
ray.init(ignore_reinit_error=True)

# Paths to the checkpoints
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

print("Loaded offensive and defensive agents from checkpoints.")
# Instantiate the hierarchical environment with the loaded agents
hierarchical_env = HierarchicalEnv()

# Example of running an episode
obs = hierarchical_env.reset()
print("random sample", hierarchical_env.action_space.sample())
done = False
total_reward = 0

#get a random action from the agent
n_steps = 50
while not done:
    action = hierarchical_env.action_space.sample()  # Sample a random action from the action space
    print("action", action)
    obs, reward, done, _, info = hierarchical_env.step(action)
    print(f"Action: {action}, Reward: {reward}, Obs: {obs}, Done: {done}")

#while not done:
    # Randomly select policy and action for demonstration (replace with proper logic)
    #policy_choice = np.random.choice([0, 1])  # 0: defensive, 1: offensive
    #action = np.random.choice([0, 1, 2, 3])  # Assuming a discrete action space with 4 actions
    # action = hierarchical_env.action_space.sample()  # Sample a random action from the action space
    # obs, reward, done, info = hierarchical_env.step(#hierarchical_env.step({"agent_1": (policy_choice, action)})
    # total_reward += reward
    # print(f"Policy: {policy_choice}, Action: {action}, Reward: {reward}, Obs: {obs}, Done: {done}")

# print(f"Total reward: {total_reward}")

import os
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from grid_env import HierarchicalEnv  # Assuming this is defined in grid_env.py

# Instantiate the hierarchical environment
model_name ="ppo_hierarchical"

hierarchical_env = HierarchicalEnv()

# Check the environment to ensure it's compatible with Stable Baselines3
check_env(hierarchical_env)

# Define the model using Stable Baselines3's PPO
#meta_agent = PPO("MlpPolicy", hierarchical_env, verbose=1)
meta_agent = PPO("MlpPolicy", 
            hierarchical_env,
            learning_rate=0.00003,
            seed=1, 
            verbose=1, 
            tensorboard_log='tensorboard_logs/')

# Define a checkpoint callback to save the model periodically
checkpoint_callback = CheckpointCallback(save_freq=10000, 
                                        save_path='./models/'+model_name+'_1/',
                                        name_prefix=model_name)

# Train the meta-agent
meta_agent.learn(total_timesteps=200000, callback=checkpoint_callback)

# Save the final model
meta_agent.save("ppo_meta_agent_final")

print("Training completed and model saved.")

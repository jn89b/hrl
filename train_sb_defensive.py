import os
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from grid_env import DefensivePolicyEnv  # Assuming this is defined in grid_env.py
from stable_baselines3.common.env_util import make_vec_env

if __name__ == "__main__":

    # Configuration
    LOAD_MODEL = True
    TOTAL_TIMESTEPS = 500000
    CONTINUE_TRAINING = True
    COMPARE_MODELS = False
    num_envs = 30  # Number of parallel environments

    # Function to create environments for parallel execution
    def make_env():
        def _init():
            env = DefensivePolicyEnv()
            return env
        return _init

    # Create a vectorized environment with 30 parallel environments
    vec_env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    vec_env = VecMonitor(vec_env)  # Monitor the vectorized environment

    # Model name and checkpoint settings
    model_name = "ppo_defensive"
    checkpoint_callback = CheckpointCallback(save_freq=10000, 
                                            save_path='./models/'+model_name+'_1/',
                                            name_prefix=model_name)

    # Load or initialize the model
    if LOAD_MODEL and not CONTINUE_TRAINING:
        print("Loading model:", model_name)
        model = PPO.load(model_name)
        model.set_env(vec_env)
        print("Model loaded.")
    elif LOAD_MODEL and CONTINUE_TRAINING:
        print("Loading model and continuing training:", model_name)
        model = PPO.load(model_name)
        model.set_env(vec_env)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, 
                    log_interval=1,
                    callback=checkpoint_callback)
        model.save(model_name)
        print("Model saved.")
    else:
        print("Initializing a new model for training.")
        model = PPO("MlpPolicy", 
                    vec_env,
                    learning_rate=0.003,
                    seed=1, 
                    verbose=1, 
                    tensorboard_log='tensorboard_logs/'+model_name, 
                    device='cuda')
        model.learn(total_timesteps=TOTAL_TIMESTEPS, 
                    log_interval=1, 
                    callback=checkpoint_callback)
        model.save(model_name)
        print("Model saved.")

    # Done with training or evaluation, cleanup resources
    done = False
    # You can now evaluate the trained model
    # obs = env.reset()

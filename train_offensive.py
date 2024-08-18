import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import BaseEnv
import gym
from gym import spaces
import numpy as np
from grid_env import OffensivePolicyEnv  # Assuming your environment is defined in grid_env.py

#create offensive policy environment


# Initialize Ray
ray.init(ignore_reinit_error=True)

# Define PPO configuration for training
ppo_config = (
    PPOConfig()
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
    .environment(env=OffensivePolicyEnv, env_config={})
    .env_runners(
        num_env_runners=1,
        num_cpus_per_env_runner=1,
    )
    .framework("torch")  # or "tf" depending on your preference
    .training(
        vf_loss_coeff=0.005,
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=30,
        lr=0.0003,
    )
    .multi_agent(
        policies={
            "default_policy": (None, OffensivePolicyEnv().observation_space, 
                               OffensivePolicyEnv().action_space, {}),
        },
        policy_mapping_fn=lambda agent_id, episode, **kwargs: 
            "default_policy",
    )
)

# Define storage path and experiment name
cwd = os.getcwd()
storage_path = os.path.join(cwd, "ray_results", "OffensivePolicyEnv")
exp_name = "PPO_OffensivePolicyEnv"

# Run the training using tune
tune.run(
    "PPO",
    name=exp_name,
    stop={"timesteps_total": 100000},
    config=ppo_config.to_dict(),
    checkpoint_freq=10,
    storage_path=storage_path,
    log_to_file=True,
)
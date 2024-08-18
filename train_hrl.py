import os 
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from grid_env import OffensivePolicyEnv, DefensivePolicyEnv, HierarchicalEnv  # Assuming these are defined in grid_env.py
from config import PWD_DEFENSIVE, PWD_OFFENSIVE  # Assuming these are defined in config.py

# Load configurations and restore checkpoints
def load_trained_agent(checkpoint_path, env_class):
    config = PPOConfig().framework("torch").environment(env=env_class)
    agent = config.build()
    agent.restore(checkpoint_path)
    return agent

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Paths to the checkpoints
trained_defensive = "PPO_DefensivePolicyEnv/trained_0"
defense_path = os.path.join(PWD_DEFENSIVE, trained_defensive+"/checkpoint_000001")

trained_offensive = "PPO_OffensivePolicyEnv/trained_0"
offense_path = os.path.join(PWD_OFFENSIVE, trained_offensive+"/checkpoint_000001")

offensive_agent = load_trained_agent(offense_path, OffensivePolicyEnv)
defensive_agent = load_trained_agent(defense_path, DefensivePolicyEnv)

print("Loaded offensive and defensive agents from checkpoints.")

# Instantiate the hierarchical environment with the loaded agents
hierarchical_env = HierarchicalEnv()
# hierarchical_env.load_agents(offensive_agent, defensive_agent)

meta_agent_config = (
    PPOConfig()
    .environment(env=HierarchicalEnv)
    .framework("torch")
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    .training(
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=30,
        lr=0.0003,
    )
)

# Define storage path and experiment name
cwd = os.getcwd()
exp_name = "PPO_MetaHierarchicalEnv"
storage_path = os.path.join(cwd, "ray_results", "MetaHierarchicalEnv")

# Run the training using tune
tune.run(
    "PPO",
    name=exp_name,
    stop={"timesteps_total": 200000},
    config=meta_agent_config.to_dict(),
    checkpoint_freq=10,
    storage_path=storage_path,
    log_to_file=True,
)
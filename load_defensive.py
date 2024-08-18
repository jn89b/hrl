import os
import ray
import matplotlib.pyplot as plt
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.dqn import DQNConfig, DQN
from ray.tune.registry import register_env
from grid_env import DefensivePolicyEnv
from config import PWD_DEFENSIVE

def env_creator(env_config=None):
    return DefensivePolicyEnv()  # return an env instance

# env_name = "uam_env"
# register_env(env_name, env_creator)

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
# ray.init(runtime_env={"working_dir": "."})
ray.init()

if __name__ == "__main__":
    trained_folder = "PPO_DefensivePolicyEnv/trained_0"
    # /home/justin/coding_projects/hrl/ray_results/OffensivePolicyEnv/PPO_OffensivePolicyEnv
    storage_path = os.path.join(PWD_DEFENSIVE, trained_folder+"/checkpoint_000001")
    #check if path exists
    if not os.path.exists(storage_path):
        raise ValueError("Checkpoint path does not exist: ", storage_path)
    
    print("Loading checkpoint from: ", storage_path)
    # # Instantiate the environment to obtain observation and action spaces
    # temp_env = OffensivePolicyEnv()

    # Load the configuration used during training
    base_config = (
        PPOConfig()
        .environment(env=DefensivePolicyEnv)
        .framework("torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .rollouts(num_envs_per_worker=2)
        .multi_agent(
            policies={
                "default_policy": (None, DefensivePolicyEnv().observation_space, 
                                DefensivePolicyEnv().action_space, {}),
            },
            policy_mapping_fn=lambda agent_id, episode, **kwargs: 
                "default_policy",
        )
    )
    
    agent = PPO(config=base_config)
    # Restore the agent from the checkpoint
    agent.restore(storage_path)
    print("Agent restored")
    
    # Now you can use the agent to run inferences in the environment
    env = DefensivePolicyEnv()
    # Reset the environment
    obs,info = env.reset()
    #set seed number
    # Run the agent in the environment
    num_success = 0
    num_trials = 1
    target_history = []
    agent_history = []
    max_steps = 30
    current_step = 0
    for i in range(num_trials):
        obs,info = env.reset()
        done = False
        num_attempts = 0
        #target_history.append(obs[2:4])

        while current_step <= max_steps:
            action = agent.compute_single_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            num_attempts += 1
            print("obs: ", obs, "action: ", action, "reward: ", reward, "done: ", done)
            agent_history.append(obs[0:2])
            target_history.append(obs[2:4])
            current_step += 1
            

    #plot the agent position and target position
    target_x = [pos[0] for pos in target_history]
    target_y = [pos[1] for pos in target_history]
    
    agent_x = [pos[0] for pos in agent_history]
    agent_y = [pos[1] for pos in agent_history]
    
    fig, ax = plt.subplots()
    #plot as a dotted line
    # ax.plot(target_x, target_y, 'r--', label='Target Position')
    ax.scatter(target_x, target_y, c='red', s=10, label='Bad Guy')  # Scatter plot for target positions
    ax.plot(target_x, target_y, 'r--', label='Bad Guy Position')

    ax.plot(agent_x, agent_y, 'b-', label='Agent Position')
    ax.scatter(agent_x, agent_y, c='blue', s=10)  # Scatter plot for agent positions
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    ax.set_title('Agent vs Bad Guy Position')
    ax.legend()
    ax.grid()
    
    plt.show()  # Show the plot 
    
    
from grid_env import DefensivePolicyEnv, OffensivePolicyEnv

from stable_baselines3 import PPO as StablePPO

defensive_name = "ppo_defensive"
offensive_name = "ppo_offensive"

offensive_env = OffensivePolicyEnv()
offensive_agent = StablePPO.load(offensive_name, env=offensive_env)
# offensive_agent.set_env(offensive_env)

defensive_env = DefensivePolicyEnv()
defensive_agent = StablePPO.load(defensive_name, env=defensive_env)
# defensive_agent.set_env(defensive_env)


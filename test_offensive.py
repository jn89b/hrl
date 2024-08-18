# test the offensive  environment
from grid_env import OffensivePolicyEnv


def test_offensive_env(N_steps=10):
    grid_env = OffensivePolicyEnv()
    
    for i in range(N_steps):
        action = grid_env.action_space.sample()
        obs, reward, done, truncated, info = grid_env.step(action)
        print(f"Step {i}:")
        print(f"  Action: {action}")
        print(f"  Observation: {obs}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        
if __name__ == "__main__":
    test_offensive_env()  # Run the test function for the OffensivePolicyEnv
            
        
        
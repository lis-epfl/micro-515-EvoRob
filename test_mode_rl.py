import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from evorob.world.envs.ant_flat import AntFlatEnvironment

# 1. Paths to your successful run files
MODEL_PATH = "/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/results/ppo_ckpts/ppo_ant_10000000_steps.zip"  # Path to your saved PPO model
STATS_PATH = "/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/results/ppo_ckpts/ppo_ant_vecnormalize_10000000_steps.pkl"

def main():
    # 2. Re-create the environment with rendering enabled
    # We use DummyVecEnv because PPO expects a vectorized environment
    def make_env():
        return AntFlatEnvironment(render_mode="human")
    
    env = DummyVecEnv([make_env])

    # 3. Load the Normalization Stats
    # This is CRITICAL. Without this, the model won't understand the observations.
    env = VecNormalize.load(STATS_PATH, env)
    
    # Disable training mode for evaluation
    env.training = False
    env.norm_reward = False

    # 4. Load the PPO Model
    print(f"Loading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH, env=env)

    # 5. The Play Loop
    obs = env.reset()
    print("Playing... Press Ctrl+C in the terminal to stop.")
    
    try:
        while True:
            # Tell the model to pick the best action (deterministic=True)
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            
            # The window will update automatically because of render_mode="human"
            if dones[0]:
                obs = env.reset()
                
    except KeyboardInterrupt:
        print("\nClosing...")
    finally:
        env.close()

if __name__ == "__main__":
    main()
import numpy as np
import gymnasium as gym
import time
from evorob.world.robot.controllers.sinoid import OscillatoryController

def visualize_best_oscillator(checkpoint_path: str):
    # 1. Load the genotype
    try:
        genotype = np.load(checkpoint_path)
        print(f"Loaded genotype from {checkpoint_path}")
        print(f"Shape: {genotype.shape}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 2. Setup the Environment (Human render mode for a window)
    # We use Ant-v5 to match the benchmark
    env = gym.make("Ant-v5", render_mode="human")
    
    # 3. Setup the Controller
    controller = OscillatoryController(output_size=8)
    controller.set_weights(genotype)

    # 4. Run Loop
    print("Starting visualization. Press Ctrl+C in terminal to stop.")
    try:
        while True:
            obs, _ = env.reset()
            controller.reset_controller()
            done = False
            truncated = False
            total_reward = 0
            
            while not (done or truncated):
                # Get rhythmic action
                action = controller.get_action(obs)
                
                # Step physics
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                # Small sleep to keep it at real-time speed (Ant is 50Hz)
                time.sleep(0.02) 
                
            print(f"Episode finished. Reward: {total_reward:.2f}")
            time.sleep(1.0) # Pause before restarting
            
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    # Point this to the x_best.npy you just uploaded
    PATH_TO_X_BEST = "results/20260318_004641_oscillatory_controller_ckpts/185/x_best.npy" 
    visualize_best_oscillator(PATH_TO_X_BEST)
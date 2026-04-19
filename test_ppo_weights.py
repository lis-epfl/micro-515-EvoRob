import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from evorob.world.envs.ant_flat import AntFlatEnvironment
from evorob.world.robot.controllers.mlp import NeuralNetworkController

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
MODEL_PATH = "/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/results/ppo_ckpts/ppo_ant_10000000_steps.zip"
STATS_PATH = "/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/results/ppo_ckpts/ppo_ant_vecnormalize_10000000_steps.pkl"


def infer_policy_architecture(model: PPO):
    hidden_sizes = []
    policy_net = model.policy.mlp_extractor.policy_net
    for layer in policy_net:
        if hasattr(layer, "weight") and hasattr(layer, "bias"):
            hidden_sizes.append(int(layer.weight.shape[0]))
    return hidden_sizes


def main():
    # 1. Re-create the environment with rendering enabled
    def make_env():
        return AntFlatEnvironment(render_mode="human")

    env = DummyVecEnv([make_env])

    # 2. Load normalization stats
    env = VecNormalize.load(STATS_PATH, env)
    env.training = False
    env.norm_reward = False

    # 3. Load PPO model
    print(f"Loading PPO model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH, device="cpu")

    # 4. Infer actor architecture and create EvoRob controller
    hidden_sizes = infer_policy_architecture(model)
    print("Inferred PPO hidden sizes:", hidden_sizes)

    controller = NeuralNetworkController(
        input_size=27,
        output_size=8,
        hidden_size=hidden_sizes,
    )
    controller.load_from_ppo_model(model)

    print("Loaded controller layer shapes:")
    for i, (W, b) in enumerate(zip(controller.weights, controller.biases)):
        print(f"  Layer {i}: W{W.shape}, b{b.shape}")
    print("Controller n_params:", controller.n_params)

    # 5. Play loop using controller.get_action()
    obs = env.reset()
    controller.reset_controller(batch_size=1)

    print("Playing PPO-loaded NeuralNetworkController... Press Ctrl+C to stop.")

    try:
        while True:
            action = controller.get_action(obs)
            obs, rewards, dones, infos = env.step(action)

            if dones[0]:
                obs = env.reset()
                controller.reset_controller(batch_size=1)

    except KeyboardInterrupt:
        print("\nClosing...")
    finally:
        env.close()


if __name__ == "__main__":
    main()
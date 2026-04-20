import os
from typing import Optional

import numpy as np

from evorob.utils.filesys import get_last_checkpoint_dir
from evorob.world.ant_world import AntFlatWorld
from evorob.world.robot.controllers.mlp_sol import NeuralNetworkController


def infer_hidden_size(genotype_len: int, input_size: int = 27, output_size: int = 8) -> int:
    """
    Infer hidden size for the biased MLP:
    total_params = input*hidden + hidden*output + hidden + output
                 = hidden*(input + output + 1) + output
    """
    numerator = genotype_len - output_size
    denominator = input_size + output_size + 1

    if numerator % denominator != 0:
        raise ValueError(
            f"Cannot infer hidden size from genotype length {genotype_len}. "
            f"Expected (len - {output_size}) divisible by {denominator}."
        )

    return numerator // denominator


def load_best_genotype(checkpoint_dir: str) -> np.ndarray:
    last_gen = get_last_checkpoint_dir(checkpoint_dir)
    x_best_path = None

    if last_gen is not None:
        candidate = os.path.join(last_gen, "x_best.npy")
        if os.path.isfile(candidate):
            x_best_path = candidate

    if x_best_path is None:
        candidate = os.path.join(checkpoint_dir, "x_best.npy")
        if os.path.isfile(candidate):
            x_best_path = candidate

    if x_best_path is None:
        raise FileNotFoundError(f"Could not find x_best.npy in '{checkpoint_dir}'")

    genotype = np.load(x_best_path)
    print(f"Loaded genotype from: {x_best_path}  (shape: {genotype.shape})")
    return genotype


def test_checkpoint(
    checkpoint_dir: str,
    n_episodes: int = 5,
    render: bool = True,
    max_steps: int = 1000,
    hidden_size: Optional[int] = None,
) -> None:
    genotype = load_best_genotype(checkpoint_dir)

    if genotype.ndim != 1:
        raise ValueError(f"Expected 1D genotype, got shape {genotype.shape}")

    if hidden_size is None:
        hidden_size = infer_hidden_size(len(genotype))

    print(f"Inferred/using hidden_size = {hidden_size}")

    world = AntFlatWorld(controller_cls=NeuralNetworkController)
    env = world.create_env(render_mode="human" if render else None)

    controller = NeuralNetworkController(
        input_size=27,
        output_size=8,
        hidden_size=hidden_size,
    )
    controller.geno2pheno(genotype)

    rewards = []

    try:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            controller.reset_controller(batch_size=1)

            total_reward = 0.0

            for step in range(max_steps):
                # obs is expected to be batched: (1, obs_dim)
                action = controller.get_action(obs)

                # Ensure vector-env action shape is (1, act_dim)
                if action.ndim == 1:
                    action = action[None, :]

                obs, reward, terminated, truncated, info = env.step(action)

                # reward/terminated/truncated are vectorized outputs
                total_reward += float(reward[0])

                if bool(terminated[0]) or bool(truncated[0]):
                    break

            rewards.append(total_reward)
            print(
                f"Episode {ep + 1}/{n_episodes} | "
                f"reward = {total_reward:.2f} | "
                f"steps = {step + 1}"
            )

    finally:
        env.close()

    print("\nSummary")
    print("-" * 40)
    print(f"Mean reward: {np.mean(rewards):.2f}")
    print(f"Std reward : {np.std(rewards):.2f}")
    print(f"Min reward : {np.min(rewards):.2f}")
    print(f"Max reward : {np.max(rewards):.2f}")


if __name__ == "__main__":
    test_checkpoint(
        checkpoint_dir="/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/results/20260419_182400_neural_controller_ckpts", #sigma 0.25
        #checkpoint_dir="/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/results/20260419_164112_neural_controller_ckpts", #    sigma 0.25  drift_penalty = 0.05 * (y_velocity ** 2) quat = self.data.qpos[3:7].copy()orientation_penalty = 0.3 * (quat[1] ** 2 + quat[2] ** 2)reward = (forward_reward+ healthy_reward- ctrl_cost- drift_penalty
        #checkpoint_dir="/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/results/20260419_153844_neural_controller_ckpts", #0.25 0.05 * (y_velocity ** 2)     reward = (forward_reward+ healthy_reward- ctrl_cost forward velocity 2 
        #checkpoint_dir="/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/results/20260419_154327_neural_controller_ckpts", #0.25 0.05 * (y_velocity ** 2)eward = (forward_reward+ healthy_reward- ctrl_cost forward velocity 
        
        n_episodes=5,
        render=True,
        max_steps=2000,
        hidden_size=None,
    )
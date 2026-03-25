import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from evorob.algorithms.ea_api import EvoAlgAPI
from evorob.world.envs.ant_flat import AntFlatEnvironment
from evorob.world.robot.controllers.mlp import NeuralNetworkController
from evorob.world.robot.controllers.rnn import (
    FrozenPPORNNResidualController,
)


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
MODEL_PATH = "/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/results/ppo_ckpts/ppo_ant_10000000_steps.zip"
STATS_PATH = "/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/results/ppo_ckpts/ppo_ant_vecnormalize_10000000_steps.pkl"


# ---------------------------------------------------------------------
# GLOBAL PPO CACHE
# ---------------------------------------------------------------------
_CACHED_HIDDEN_SIZES = None
_CACHED_PPO_WEIGHTS = None
_CACHED_PPO_BIASES = None


def infer_policy_architecture(model: PPO) -> List[int]:
    """
    Infer hidden-layer sizes directly from the PPO actor policy network.
    """
    hidden_sizes = []
    policy_net = model.policy.mlp_extractor.policy_net

    for layer in policy_net:
        if hasattr(layer, "weight") and hasattr(layer, "bias"):
            hidden_sizes.append(int(layer.weight.shape[0]))

    return hidden_sizes


def build_frozen_ppo_backbone(model_path: str, verbose: bool = True) -> Tuple[List[int], NeuralNetworkController]:
    """
    Load PPO once, infer architecture, and cache the copied PPO weights.
    """
    global _CACHED_HIDDEN_SIZES, _CACHED_PPO_WEIGHTS, _CACHED_PPO_BIASES

    if (
        _CACHED_HIDDEN_SIZES is not None
        and _CACHED_PPO_WEIGHTS is not None
        and _CACHED_PPO_BIASES is not None
    ):
        controller = NeuralNetworkController(
            input_size=27,
            output_size=8,
            hidden_size=_CACHED_HIDDEN_SIZES,
        )
        controller.weights = [w.copy().astype(np.float32) for w in _CACHED_PPO_WEIGHTS]
        controller.biases = [b.copy().astype(np.float32) for b in _CACHED_PPO_BIASES]
        controller.n_params = controller.get_num_params()
        return _CACHED_HIDDEN_SIZES, controller

    model = PPO.load(model_path)

    hidden_sizes = infer_policy_architecture(model)
    controller = NeuralNetworkController(
        input_size=27,
        output_size=8,
        hidden_size=hidden_sizes,
    )
    controller.load_from_ppo_model(model)

    _CACHED_HIDDEN_SIZES = hidden_sizes
    _CACHED_PPO_WEIGHTS = [w.copy().astype(np.float32) for w in controller.weights]
    _CACHED_PPO_BIASES = [b.copy().astype(np.float32) for b in controller.biases]

    if verbose:
        print("Inferred PPO hidden sizes:", hidden_sizes)
        print("Loaded PPO backbone layer shapes:")
        for i, (W, b) in enumerate(zip(controller.weights, controller.biases)):
            print(f"  Layer {i}: W{W.shape}, b{b.shape}")
        print("Frozen PPO controller n_params:", controller.n_params)

    return hidden_sizes, controller


def clone_frozen_ppo_controller() -> Tuple[List[int], NeuralNetworkController]:
    """
    Clone the cached frozen PPO backbone without reloading PPO from disk.
    """
    hidden_sizes, cached_controller = build_frozen_ppo_backbone(MODEL_PATH, verbose=False)

    controller = NeuralNetworkController(
        input_size=27,
        output_size=8,
        hidden_size=hidden_sizes,
    )
    controller.weights = [w.copy().astype(np.float32) for w in cached_controller.weights]
    controller.biases = [b.copy().astype(np.float32) for b in cached_controller.biases]
    controller.n_params = controller.get_num_params()

    return hidden_sizes, controller


def make_env():
    return AntFlatEnvironment(render_mode=None)


def make_vecnormalize_env():
    """
    Recreate the same normalized evaluation setup used by PPO evaluation.
    """
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(STATS_PATH, env)
    env.training = False
    env.norm_reward = False
    return env


def make_initial_residual_genotype(
    hidden_size: int,
    input_size: int = 27,
    output_size: int = 8,
) -> np.ndarray:
    """
    Create a near-zero residual initialization so the initial controller behaves
    almost exactly like frozen PPO.

    Layout expected by FrozenPPORNNResidualController.set_weights():
      [Wxh | Whh | bh | Why | by]
    """
    rng = np.random.default_rng(42)

    Wxh = rng.normal(0.0, 0.01, size=(hidden_size, input_size)).astype(np.float32)
    Whh = rng.normal(0.0, 0.01, size=(hidden_size, hidden_size)).astype(np.float32)
    bh = np.zeros(hidden_size, dtype=np.float32)

    # Keep residual output very small initially
    Why = np.zeros((output_size, hidden_size), dtype=np.float32)
    by = np.zeros(output_size, dtype=np.float32)

    return np.concatenate([
        Wxh.reshape(-1),
        Whh.reshape(-1),
        bh.reshape(-1),
        Why.reshape(-1),
        by.reshape(-1),
    ]).astype(np.float32)


def build_residual_controller_from_genotype(
    hidden_size: int,
    residual_scale: float,
    genotype: np.ndarray,
) -> FrozenPPORNNResidualController:
    """
    Build frozen PPO backbone + trainable RNN residual and set only
    the residual weights from genotype.
    """
    _, frozen_ppo = clone_frozen_ppo_controller()

    controller = FrozenPPORNNResidualController(
        ppo_controller=frozen_ppo,
        input_size=27,
        output_size=8,
        hidden_size=hidden_size,
        residual_scale=residual_scale,
    )
    controller.set_weights(genotype)
    return controller


def evaluate_residual_individual_with_rl_normalization(
    genotype: np.ndarray,
    hidden_size: int = 32,
    residual_scale: float = 0.2,
    n_episodes: int = 1,
    max_episode_steps: int = 1000,
    seed: int = 0,
) -> float:
    """
    Evaluate frozen PPO + trainable RNN residual.
    """
    controller = build_residual_controller_from_genotype(
        hidden_size=hidden_size,
        residual_scale=residual_scale,
        genotype=genotype,
    )

    vec_env = make_vecnormalize_env()
    rng = np.random.default_rng(seed)
    episode_rewards = []

    for _ in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))

        obs = vec_env.reset()
        try:
            vec_env.env_method("reset", seed=ep_seed)
            obs = vec_env.reset()
        except Exception:
            obs = vec_env.reset()

        controller.reset_controller(batch_size=1)
        total_reward = 0.0

        for _ in range(max_episode_steps):
            action = controller.get_action(obs)
            obs, rewards, dones, infos = vec_env.step(action)

            total_reward += float(rewards[0])

            if dones[0]:
                break

        episode_rewards.append(total_reward)

    vec_env.close()
    return float(np.mean(episode_rewards))


def evaluate_frozen_ppo_baseline(
    n_episodes: int = 1,
    max_episode_steps: int = 1000,
    seed: int = 0,
) -> float:
    """
    Evaluate frozen PPO baseline alone, for reference.
    """
    _, ppo_controller = clone_frozen_ppo_controller()

    vec_env = make_vecnormalize_env()
    rng = np.random.default_rng(seed)
    episode_rewards = []

    for _ in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))

        obs = vec_env.reset()
        try:
            vec_env.env_method("reset", seed=ep_seed)
            obs = vec_env.reset()
        except Exception:
            obs = vec_env.reset()

        ppo_controller.reset_controller(batch_size=1)
        total_reward = 0.0

        for _ in range(max_episode_steps):
            action = ppo_controller.get_action(obs)
            obs, rewards, dones, infos = vec_env.step(action)

            total_reward += float(rewards[0])

            if dones[0]:
                break

        episode_rewards.append(total_reward)

    vec_env.close()
    return float(np.mean(episode_rewards))


def run_evolution_frozen_ppo_rnn_residual(
    num_generations: int,
    population_size: int,
    ckpt_interval: int,
    checkpoint_path: Optional[str] = None,
    random_seed: int = 42,
    sigma: float = 0.02,
    hidden_size: int = 32,
    residual_scale: float = 0.2,
    n_eval_episodes: int = 1,
    max_episode_steps: int = 1000,
) -> None:
    """
    Run CMA-ES on the residual RNN module only.
    PPO backbone remains frozen.
    """
    np.random.seed(random_seed)

    # Preload frozen PPO once
    build_frozen_ppo_backbone(MODEL_PATH, verbose=True)

    x0 = make_initial_residual_genotype(hidden_size=hidden_size)
    print("Residual genotype length:", x0.shape[0])

    print("\nEvaluating frozen PPO baseline...")
    ppo_fitness = evaluate_frozen_ppo_baseline(
        n_episodes=n_eval_episodes,
        max_episode_steps=max_episode_steps,
        seed=random_seed,
    )
    print(f"Frozen PPO baseline fitness: {ppo_fitness:.2f}")

    print("\nEvaluating exact initial residual individual (near-zero residual)...")
    x0_fitness = evaluate_residual_individual_with_rl_normalization(
        genotype=x0,
        hidden_size=hidden_size,
        residual_scale=residual_scale,
        n_episodes=n_eval_episodes,
        max_episode_steps=max_episode_steps,
        seed=random_seed,
    )
    print(f"Initial residual fitness: {x0_fitness:.2f}")

    dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if checkpoint_path is None:
        checkpoint_path = f"results/{dt_str}_frozen_ppo_rnn_residual_ckpts"
    else:
        checkpoint_path = str(
            Path(checkpoint_path).parent / f"{dt_str}_{Path(checkpoint_path).name}"
        )

    ckpt_dir = Path(checkpoint_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ea = EvoAlgAPI(
        n_params=x0.shape[0],
        population_size=population_size,
        num_generations=num_generations,
        output_dir=ckpt_dir,
        sigma=sigma,
        x0=x0,
    )

    best_fitness = -np.inf
    best_individual = None

    print("\n" + "=" * 72)
    print("EVOLVING TRAINABLE RNN RESIDUAL ON TOP OF FROZEN PPO")
    print("=" * 72)
    print(f"Frozen PPO baseline : {ppo_fitness:.2f}")
    print(f"Initial residual x0 : {x0_fitness:.2f}")
    print(f"Residual hidden size: {hidden_size}")
    print(f"Residual scale      : {residual_scale}")
    print("=" * 72)

    for generation in range(num_generations):
        population = ea.ask()

        # Force the exact near-zero residual into generation 1
        if generation == 0:
            population[0] = x0.copy()

        fitness = np.empty(len(population), dtype=np.float32)

        for i, individual in enumerate(population):
            fitness[i] = evaluate_residual_individual_with_rl_normalization(
                genotype=individual,
                hidden_size=hidden_size,
                residual_scale=residual_scale,
                n_episodes=n_eval_episodes,
                max_episode_steps=max_episode_steps,
                seed=random_seed + generation * 1000 + i,
            )

        save_checkpoint = (generation % ckpt_interval == 0) or (
            generation == num_generations - 1
        )
        ea.tell(population, fitness, save_checkpoint=save_checkpoint)

        gen_best_idx = int(np.argmax(fitness))
        gen_best_fitness = float(fitness[gen_best_idx])
        gen_mean_fitness = float(np.mean(fitness))
        gen_median_fitness = float(np.median(fitness))

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = population[gen_best_idx].copy()

        x0_tag = ""
        if generation == 0:
            x0_tag = f" | x0 residual fitness={fitness[0]:.2f}"

        print(
            f"Generation {generation + 1}/{num_generations}: "
            f"Best={gen_best_fitness:.2f}, "
            f"Mean={gen_mean_fitness:.2f}, "
            f"Median={gen_median_fitness:.2f}, "
            f"Overall Best={ea.f_best_so_far:.2f}"
            f"{x0_tag}"
        )

    print("\nDone.")
    print(f"Best fitness found: {best_fitness:.2f}")
    print(f"Checkpoints saved to: {ckpt_dir}")

    if best_individual is not None:
        best_path = ckpt_dir / "x_best_rnn_residual.npy"
        np.save(best_path, best_individual)
        print(f"Saved best residual individual to: {best_path}")

    np.save(ckpt_dir / "x0_rnn_residual.npy", x0)
    print(f"Saved initial residual individual to: {ckpt_dir / 'x0_rnn_residual.npy'}")


if __name__ == "__main__":
    run_evolution_frozen_ppo_rnn_residual(
        num_generations=1000,
        population_size=300,
        ckpt_interval=5,
        checkpoint_path=None,
        random_seed=42,
        sigma=0.5,
        hidden_size=32,
        residual_scale=0.2,
        n_eval_episodes=1,
        max_episode_steps=1000,
    )
import datetime
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from evorob.algorithms.ea_api import EvoAlgAPI
from evorob.world.envs.ant_flat import AntFlatEnvironment
from evorob.world.robot.controllers.mlp import NeuralNetworkController


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
MODEL_PATH = "/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/results/ppo_ckpts/ppo_ant_10000000_steps.zip"
STATS_PATH = "/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/results/ppo_ckpts/ppo_ant_vecnormalize_10000000_steps.pkl"


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


def flatten_controller_params(controller: NeuralNetworkController) -> np.ndarray:
    """
    Flatten controller params in the same order expected by set_weights():
    [W1, b1, W2, b2, ..., Wn, bn]
    """
    flat_parts = []
    for W, b in zip(controller.weights, controller.biases):
        flat_parts.append(W.reshape(-1))
        flat_parts.append(b.reshape(-1))
    return np.concatenate(flat_parts).astype(np.float32)


def build_rl_controller(model_path: str) -> Tuple[List[int], NeuralNetworkController]:
    """
    Load PPO, infer architecture, and copy PPO weights into our controller.
    """
    model = PPO.load(model_path)

    hidden_sizes = infer_policy_architecture(model)
    print("Inferred PPO hidden sizes:", hidden_sizes)

    controller = NeuralNetworkController(
        input_size=27,
        output_size=8,
        hidden_size=hidden_sizes,
    )
    controller.load_from_ppo_model(model)

    print("Loaded PPO layer shapes:")
    for i, (W, b) in enumerate(zip(controller.weights, controller.biases)):
        print(f"  Layer {i}: W{W.shape}, b{b.shape}")

    print("Controller n_params      :", controller.n_params)
    print("Flattened genotype length:", flatten_controller_params(controller).shape[0])

    return hidden_sizes, controller


def extract_frozen_and_last_layer(
    model_path: str,
) -> Tuple[List[int], List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Load PPO and split controller params into:
    - frozen layers: all hidden layers
    - last-layer genotype: final action layer only
    """
    hidden_sizes, controller = build_rl_controller(model_path)

    frozen_weights = [w.copy().astype(np.float32) for w in controller.weights[:-1]]
    frozen_biases = [b.copy().astype(np.float32) for b in controller.biases[:-1]]

    last_W = controller.weights[-1].copy().astype(np.float32)
    last_b = controller.biases[-1].copy().astype(np.float32)

    x0_last = np.concatenate([
        last_W.reshape(-1),
        last_b.reshape(-1),
    ]).astype(np.float32)

    print("Frozen PPO layers:")
    for i, (W, b) in enumerate(zip(frozen_weights, frozen_biases)):
        print(f"  Frozen layer {i}: W{W.shape}, b{b.shape}")

    print("Evolved last layer:")
    print(f"  W_last{last_W.shape}, b_last{last_b.shape}")
    print(f"  Last-layer genotype length: {x0_last.shape[0]}")

    return hidden_sizes, frozen_weights, frozen_biases, x0_last


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


def set_controller_last_layer(
    controller: NeuralNetworkController,
    frozen_weights: List[np.ndarray],
    frozen_biases: List[np.ndarray],
    last_layer_genotype: np.ndarray,
) -> None:
    """
    Rebuild a full controller from:
    - frozen hidden layers
    - evolved last layer
    """
    out_dim, in_dim = controller.weights[-1].shape
    w_size = out_dim * in_dim
    b_size = out_dim

    g = np.asarray(last_layer_genotype, dtype=np.float32).ravel()
    expected = w_size + b_size
    if len(g) != expected:
        raise ValueError(
            f"Last-layer genotype length mismatch: expected {expected}, got {len(g)}"
        )

    W_last = g[:w_size].reshape(out_dim, in_dim).astype(np.float32)
    b_last = g[w_size:w_size + b_size].astype(np.float32)

    controller.weights = [w.copy().astype(np.float32) for w in frozen_weights] + [W_last]
    controller.biases = [b.copy().astype(np.float32) for b in frozen_biases] + [b_last]
    controller.n_params = controller.get_num_params()


def evaluate_last_layer_individual_with_rl_normalization(
    last_layer_genotype: np.ndarray,
    hidden_sizes: List[int],
    frozen_weights: List[np.ndarray],
    frozen_biases: List[np.ndarray],
    n_episodes: int = 1,
    max_episode_steps: int = 1000,
    seed: int = 0,
) -> float:
    """
    Evaluate a controller where only the final layer is evolved.
    Hidden layers remain frozen from PPO.
    """
    controller = NeuralNetworkController(
        input_size=27,
        output_size=8,
        hidden_size=hidden_sizes,
    )

    set_controller_last_layer(
        controller=controller,
        frozen_weights=frozen_weights,
        frozen_biases=frozen_biases,
        last_layer_genotype=last_layer_genotype,
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


def run_evolution_last_layer_rl_warmstart(
    num_generations: int,
    population_size: int,
    ckpt_interval: int,
    checkpoint_path: Optional[str] = None,
    random_seed: int = 42,
    sigma: float = 0.01,
    n_eval_episodes: int = 1,
    max_episode_steps: int = 1000,
) -> None:
    """
    Run CMA-ES on PPO's last action layer only.
    Earlier layers remain frozen.
    """
    np.random.seed(random_seed)

    hidden_sizes, frozen_weights, frozen_biases, x0_last = extract_frozen_and_last_layer(
        MODEL_PATH
    )

    print("\nEvaluating exact PPO last-layer individual with VecNormalize stats...")
    ppo_fitness = evaluate_last_layer_individual_with_rl_normalization(
        last_layer_genotype=x0_last,
        hidden_sizes=hidden_sizes,
        frozen_weights=frozen_weights,
        frozen_biases=frozen_biases,
        n_episodes=n_eval_episodes,
        max_episode_steps=max_episode_steps,
        seed=random_seed,
    )
    print(f"Exact PPO initial fitness: {ppo_fitness:.2f}")

    dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if checkpoint_path is None:
        checkpoint_path = f"results/{dt_str}_rl_last_layer_ckpts"
    else:
        checkpoint_path = str(
            Path(checkpoint_path).parent / f"{dt_str}_{Path(checkpoint_path).name}"
        )

    ckpt_dir = Path(checkpoint_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ea = EvoAlgAPI(
        n_params=x0_last.shape[0],
        population_size=population_size,
        num_generations=num_generations,
        output_dir=ckpt_dir,
        sigma=sigma,
        x0=x0_last,
    )

    best_fitness = -np.inf
    best_individual = None

    for generation in range(num_generations):
        population = ea.ask()

        # Force exact PPO last layer into generation 1
        if generation == 0:
            population[0] = x0_last.copy()

        fitness = np.empty(len(population), dtype=np.float32)

        for i, individual in enumerate(population):
            fitness[i] = evaluate_last_layer_individual_with_rl_normalization(
                last_layer_genotype=individual,
                hidden_sizes=hidden_sizes,
                frozen_weights=frozen_weights,
                frozen_biases=frozen_biases,
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

        ppo_tag = ""
        if generation == 0:
            ppo_tag = f" | PPO individual fitness={fitness[0]:.2f}"

        print(
            f"Generation {generation + 1}/{num_generations}: "
            f"Best={gen_best_fitness:.2f}, "
            f"Mean={gen_mean_fitness:.2f}, "
            f"Median={gen_median_fitness:.2f}, "
            f"Overall Best={ea.f_best_so_far:.2f}"
            f"{ppo_tag}"
        )

    print("\nDone.")
    print(f"Best fitness found: {best_fitness:.2f}")
    print(f"Checkpoints saved to: {ckpt_dir}")

    if best_individual is not None:
        best_path = ckpt_dir / "x_best_last_layer.npy"
        np.save(best_path, best_individual)
        print(f"Saved best last-layer individual to: {best_path}")


if __name__ == "__main__":
    run_evolution_last_layer_rl_warmstart(
        num_generations=300,
        population_size=50,
        ckpt_interval=5,
        checkpoint_path=None,
        random_seed=42,
        sigma=0.001,
        n_eval_episodes=1,
        max_episode_steps=1000,
    )
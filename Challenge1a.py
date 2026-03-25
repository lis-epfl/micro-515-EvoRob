import datetime
import os
from pathlib import Path
from typing import Optional

import gymnasium as gym
#import imageio
import numpy as np

from evorob.algorithms.ea_api import EvoAlgAPI
from evorob.utils.filesys import get_last_checkpoint_dir
from evorob.world.ant_world import AntFlatWorld
from evorob.world.robot.controllers.mlp import NeuralNetworkController

"""
    Controller optimisation: Ant flat terrain
"""


def test_exercise_implementation():
    print("\n" + "=" * 60)
    print("EXERCISE 1a: Testing Components")
    print("=" * 60)

    # Test 1: Environment
    print("\n[1/3] Testing Ant Environment...")
    try:
        from evorob.world.envs.ant_flat import AntFlatEnvironment

        env = AntFlatEnvironment()

        # Test _get_obs(): should concatenate qpos[2:] and qvel (27 dims total)
        obs, _ = env.reset()
        expected_obs_size = (
            env.data.qpos.size - 2
        ) + env.data.qvel.size  # 27 = 13 + 14
        assert obs.shape[0] == expected_obs_size, (
            f"Observation should be qpos[2:]({env.data.qpos.size - 2}) + qvel"
            f"({env.data.qvel.size}) = {expected_obs_size}, got {obs.shape[0]}"
        )

        # Test 
        # (): should return (reward, reward_info_dict) with three components
        action = np.zeros(env.action_space.shape[0])  # zero action
        obs, reward, terminated, truncated, info = env.step(action)
        assert "reward_forward" in info, "Missing reward_forward - check _get_rew()"
        assert "reward_ctrl" in info, "Missing reward_ctrl - check _get_rew()"
        assert "reward_survive" in info, "Missing reward_survive - check _get_rew()"
        assert info["reward_survive"] == 1.0, "Healthy reward should be 1.0"
        assert info["reward_ctrl"] <= 0, "Control cost should be negative or zero"

        # Test _get_termination(): should check torso height and state validity
        env.reset()
        # Set torso too low (should terminate)
        qpos = env.data.qpos.copy()
        qpos[2] = 0.2  # Below 0.26 threshold
        env.set_state(qpos, env.data.qvel.copy())
        terminated_low = env._get_termination()
        assert terminated_low, "Should terminate when torso height < 0.26"

        # Set torso too high (should terminate)
        qpos[2] = 1.5  # Above 1.0 threshold
        env.set_state(qpos, env.data.qvel.copy())
        terminated_high = env._get_termination()
        assert terminated_high, "Should terminate when torso height > 1.0"

        # Set torso at healthy height (should not terminate)
        qpos[2] = 0.5  # Between 0.26 and 1.0
        env.set_state(qpos, env.data.qvel.copy())
        terminated_healthy = env._get_termination()
        assert not terminated_healthy, "Should NOT terminate when 0.26 < torso < 1.0"

        env.close()
        print("✅ Environment works correctly!")
    except NotImplementedError as e:
        print(f"❌ Not implemented: {str(e)}")
        print(
            "   👉 Implement _get_obs(), _get_rew(), _get_termination() in ant_flat.py"
        )
        exit(1)
    except AssertionError as e:
        print(f"❌ Assertion failed: {str(e)}")
        exit(1)
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
        exit(1)

    # Test 2: Neural Network Controller
    print("\n[2/3] Testing Neural Network Controller...")
    try:
        controller = NeuralNetworkController(
            input_size=27, output_size=8, hidden_size=16
        )
        test_obs = np.random.randn(27)
        actions = controller.get_action(test_obs)
        assert actions.shape == (8,), (
            f"Action shape should be (8,), got {actions.shape}"
        )
        assert np.all(actions >= -1.0) and np.all(actions <= 1.0), (
            "Actions outside [-1, 1]"
        )

        test_weights = np.random.uniform(-1, 1, controller.n_params)
        controller.set_weights(test_weights)
        actions_after = controller.get_action(test_obs)
        assert actions_after.shape == (8,), "Actions shape changed after set_weights"
        print("✅ Neural Network Controller works correctly!")
    except NotImplementedError as e:
        print(f"❌ Not implemented: {str(e)}")
        print(
            "   👉 Implement __init__(), get_action(), set_weights(), get_num_params() in mlp.py"
        )
        exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        exit(1)

    # Test 3: Evolutionary Algorithm
    print("\n[3/3] Testing Evolutionary Algorithm API...")
    try:
        ea = EvoAlgAPI(n_params=100, population_size=20, sigma=0.5)
        population = ea.ask()
        assert population.shape == (20, 100), (
            f"Population shape should be (20, 100), got {population.shape}"
        )

        fitnesses = np.random.randn(20)
        ea.tell(population, fitnesses, save_checkpoint=False)
        print("✅ Evolutionary Algorithm works correctly!")
    except NotImplementedError as e:
        print(f"❌ Not implemented: {str(e)}")
        print("   👉 Implement __init__(), ask(), tell() in ea_api.py")
        print("   💡 Tip: pip install cma, then use cma.CMAEvolutionStrategy")
        exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        exit(1)

    print("\n" + "=" * 60)
    print("🎉 All tests passed! Ready to run evolution.")
    print("=" * 60)
    print("\nUncomment the line below to start evolutionary training.")


def plot_fitness(full_f, output_dir):
    """Save a fitness-over-generations plot to the checkpoint directory."""
    import matplotlib.pyplot as plt

    fitness_array = np.array(full_f)  # (n_generations, n_pop)
    generations = np.arange(1, len(fitness_array) + 1)

    best_per_gen = np.max(fitness_array, axis=1)
    mean_per_gen = np.mean(fitness_array, axis=1)
    std_per_gen = np.std(fitness_array, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        generations,
        best_per_gen,
        label="Best",
        color="#B51F1F",
        linewidth=2,
        linestyle="--",
    )
    ax.plot(
        generations,
        mean_per_gen,
        label="Mean",
        color="#007480",
        linewidth=2,
    )
    ax.fill_between(
        generations,
        mean_per_gen - std_per_gen,
        mean_per_gen + std_per_gen,
        alpha=0.2,
        color="#007480",
        label="Mean +/- 1 std",
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness over Generations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(output_dir, "fitness_plot.pdf")
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Fitness plot saved to: {plot_path}")


def run_evolution_neural_controller(
    num_generations: int,
    population_size: int,
    ckpt_interval: int,
    checkpoint_path: Optional[str] = None,
    run_evaluation: bool = True,
    compute_score: bool = True,
    random_seed: int = 42,
) -> None:
    """Run evolutionary optimization for robot controller."""
    np.random.seed(random_seed)

    # Create world for evaluation
    world = AntFlatWorld(controller_cls=NeuralNetworkController)

    # Timestamped checkpoint directory
    dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if checkpoint_path is None:
        checkpoint_path = f"results/{dt_str}_neural_controller_ckpts"
    else:
        # If path is relative or absolute, just add prefix
        checkpoint_path = str(
            Path(checkpoint_path).parent / f"{dt_str}_{Path(checkpoint_path).name}"
        )

    ckpt_dir = Path(checkpoint_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Create evolutionary algorithm with checkpointing
    num_params = world.n_params
    ea = EvoAlgAPI(
        num_params, population_size=population_size, sigma=0.5, output_dir=ckpt_dir
    )

    # Evolution loop (checkpointing happens automatically in ea.tell())
    for generation in range(num_generations):
        # Ask EA for new population
        population = ea.ask()
        fitness = np.empty(len(population))

        for i, individual in enumerate(population):
            fitness[i] = world.evaluate_individual(individual)

        # Tell EA the results
        save_checkpoint = (generation % ckpt_interval == 0) or (
            generation == num_generations - 1
        )
        ea.tell(population, fitness, save_checkpoint=save_checkpoint)

        # Logging metrics
        gen_best_idx = np.argmax(fitness)
        gen_best_fitness = fitness[gen_best_idx]
        mean_fitness = np.mean(fitness)
        print(
            f"Generation {generation + 1}/{num_generations}: "
            f"Best={gen_best_fitness:.2f}, Mean={mean_fitness:.2f}, "
            f"Overall Best={ea.f_best_so_far:.2f}"
        )

    # Get best individual for evaluation from EA's tracking
    best_individual = ea.x_best_so_far

    print(f"\nEvolution complete! Best fitness: {ea.f_best_so_far:.2f}")
    print(f"Checkpoints saved to {ckpt_dir}")

    # Save fitness plot
    plot_fitness(ea.full_f, ckpt_dir)

    # Compute score Ant-v5 benchmark environment
    if compute_score:
        evaluate_checkpoint(
            checkpoint_dir=str(ckpt_dir),
            output_dir=str(ckpt_dir),
        )

    if run_evaluation:
        # Evaluate the trained agent with the same env factory as training
        evaluation_env = world.create_env(render_mode="human")

        evaluation_controller = world.controller
        evaluation_controller.geno2pheno(best_individual)

        obs, _ = evaluation_env.reset()
        trial_reward = 0.0
        trial_count = 0

        print("Press Ctrl+C to stop the evaluation...")
        try:
            while True:
                action = evaluation_controller.get_action(obs)
                obs, reward, terminated, truncated, _ = evaluation_env.step(action)
                trial_reward += reward

                if np.logical_or(terminated, truncated):
                    trial_count += 1
                    print(trial_reward)
                    print(f"Trial {trial_count} reward: {float(trial_reward[0]):.2f}")
                    trial_reward = 0.0
                    obs, _ = evaluation_env.reset()

        except KeyboardInterrupt:
            print(f"\n\nEvaluation stopped by user after {trial_count} trials.")
        finally:
            evaluation_env.close()


def evaluate_checkpoint(
    checkpoint_dir: str,
    output_dir: str = "evaluation_output",
) -> None:
    """Evaluate a checkpoint on the standard Gymnasium Ant-v5 (no contact forces).

    Loads the best genotype from the checkpoint, runs it for multiple episodes,
    writes a score file and records a video.

    Args:
        checkpoint_dir: Path to your EA checkpoint folder
                        (e.g. "results/20260301_120000_neural_controller_ckpts")
        output_dir:     Where to save score file and video (default: "evaluation_output")
    """
    n_episodes: int = 256  # DO NOT CHANGE!
    max_episode_steps: int = 1000  # DO NOT CHANGE!
    seed: int = 0  # DO NOT CHANGE!

    # --- Load best genotype from checkpoint ---
    last_gen = get_last_checkpoint_dir(checkpoint_dir)
    x_best_path = os.path.join(last_gen, "x_best.npy") if last_gen else ""

    if not os.path.isfile(x_best_path):
        x_best_path = os.path.join(checkpoint_dir, "x_best.npy")

    if not os.path.isfile(x_best_path):
        print(f"ERROR: Could not find x_best.npy in '{checkpoint_dir}'.")
        print("Make sure the path points to your checkpoint folder.")
        return

    genotype = np.load(x_best_path)
    print(f"Loaded genotype from: {x_best_path}  (shape: {genotype.shape})")

    # --- Create controller (same one used during training) ---
    controller = NeuralNetworkController(input_size=27, output_size=8, hidden_size=16)
    controller.geno2pheno(genotype)
    print(
        f"Controller: NeuralNetworkController  |  Parameters: {controller.n_params}\n"
    )

    # --- Run evaluation episodes on the real Ant-v5 ---
    env = gym.make(
        "Ant-v5",
        include_cfrc_ext_in_observation=False,
        max_episode_steps=max_episode_steps,
    )
    rng = np.random.default_rng(seed)
    episode_rewards = []

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=ep_seed)
        controller.reset_controller(batch_size=1)

        total_reward = 0.0
        done = False
        for _ in range(max_episode_steps):
            action = controller.get_action(obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if done:
                break

        episode_rewards.append(total_reward)
        print(f"  Episode {ep + 1}/{n_episodes}: reward = {total_reward:.2f}")

    env.close()

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    print(f"\nMean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    """

    # --- Record video ---
    print("\nRecording video...")
    video_env = gym.make(
        "Ant-v5",
        include_cfrc_ext_in_observation=False,
        max_episode_steps=max_episode_steps,
        render_mode="rgb_array",
    )
    obs, _ = video_env.reset(seed=seed)
    controller.reset_controller(batch_size=1)
    frames = []
    done = False
    video_reward = 0.0

    for _ in range(max_episode_steps):
        frames.append(video_env.render())
        action = controller.get_action(obs)
        if action.ndim > 1:
            action = action.squeeze(0)
        obs, reward, terminated, truncated, _ = video_env.step(action)
        video_reward += reward
        done = terminated or truncated

        if done:
            break

    video_env.close()

    # --- Save outputs ---
    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.join(output_dir, "evaluation_video.mp4")
    imageio.mimwrite(video_path, frames, fps=20)
    print(f"Video saved to: {video_path}")

    score_path = os.path.join(output_dir, "evaluation_score.txt")"""
    with open(score_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("MICRO-515 Challenge 1a - Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Controller type : NeuralNetworkController\n")
        f.write(f"Checkpoint      : {checkpoint_dir}\n")
        f.write(f"Environment     : Ant-v5 (no contact forces)\n")
        f.write(f"Episodes        : {n_episodes}\n\n")
        f.write("-" * 50 + "\n")
        f.write("Per-episode rewards:\n")
        for i, r in enumerate(episode_rewards):
            f.write(f"  Episode {i + 1:3d}: {r:10.2f}\n")
        f.write("-" * 50 + "\n\n")
        f.write(f"MEAN SCORE : {mean_reward:.2f}\n")
        f.write(f"STD        : {std_reward:.2f}\n")
        f.write(f"MIN        : {min(episode_rewards):.2f}\n")
        f.write(f"MAX        : {max(episode_rewards):.2f}\n\n")
        f.write(f"Video episode reward: {video_reward:.2f}\n")

    print(f"Score saved to : {score_path}")
    print(f"\n{'=' * 50}")
    print(f"  FINAL SCORE: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    test_exercise_implementation()

    # Uncomment to run full evolution:
    run_evolution_neural_controller(
        num_generations=300,
        population_size=280,
        ckpt_interval=5,
        checkpoint_path=None,
        run_evaluation=True,
        compute_score=True,
        random_seed=42,
    )

    # ----------------------------------------------------------------
    # EVALUATION: Uncomment the lines below to evaluate your checkpoint
    # on the standard Gymnasium Ant-v5 and get your final score + video.
    # Replace the path with your actual checkpoint folder.
    # ----------------------------------------------------------------
    evaluate_checkpoint(
     checkpoint_dir="results/20260304_174619_neural_controller_ckpts",
     )
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from evorob.algorithms.nsga_sol import NSGAII
from evorob.world.ant_multi_world import AntMultiWorld
from evorob.world.ant_world import AntFlatWorld
from evorob.world.robot.controllers.mlp_sol import NeuralNetworkController

""" 
    Multi-objective optimisation: Ant two-terrains
"""

def test_exercise_implementation():
    """Test NSGA-II implementation components."""
    print("\n" + "=" * 60)
    print("EXERCISE 2: Testing NSGA-II Components")
    print("=" * 60)

    # Test 1: Dominance Function
    print("\n[1/5] Testing Pareto Dominance...")
    try:
        nsga = NSGAII(population_size=10, n_opt_params=5)

        # Test case 1: Clear dominance
        assert nsga.dominates([5, 3], [4, 2]) == True, (
            "[5,3] should dominate [4,2] (better in both)"
        )

        # Test case 2: No dominance (trade-off)
        assert nsga.dominates([5, 2], [4, 3]) == False, (
            "[5,2] should NOT dominate [4,3] (trade-off)"
        )

        # Test case 3: Equal in one, better in other
        assert nsga.dominates([5, 3], [5, 2]) == True, (
            "[5,3] should dominate [5,2] (equal in first, better in second)"
        )

        # Test case 4: Identical solutions
        assert nsga.dominates([4, 3], [4, 3]) == False, (
            "[4,3] should NOT dominate [4,3] (identical)"
        )

        print("✅ Dominance function works correctly!")
    except NotImplementedError as e:
        print(f"❌ Not implemented: {str(e)}")
        print("   👉 Implement dominates() in nsga.py")
        print("   See Exercise 2a in challenge2.md for guidance")
        exit(1)
    except AssertionError as e:
        print(f"❌ Assertion failed: {str(e)}")
        print("   👉 Check your dominance logic")
        exit(1)
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
        exit(1)

    # Test 2: Fast Non-Dominated Sorting
    print("\n[2/5] Testing Fast Non-Dominated Sorting...")
    try:
        nsga = NSGAII(population_size=10, n_opt_params=5)

        # Create test fitness with known Pareto structure
        # Front 0: [5,5], [6,4], [4,6]  (non-dominated)
        # Front 1: [5,3], [3,5]  (dominated by Front 0)
        # Front 2: [3,3]  (dominated by Front 1)
        test_fitness = np.array(
            [
                [5, 5],  # Front 0
                [6, 4],  # Front 0
                [4, 6],  # Front 0
                [5, 3],  # Front 1
                [3, 5],  # Front 1
                [3, 3],  # Front 2
            ]
        )

        fronts, ranks = nsga.fast_nondominated_sort(test_fitness)

        # Verify Front 0 contains non-dominated solutions
        assert len(fronts[0]) == 3, (
            f"Front 0 should have 3 solutions, got {len(fronts[0])}"
        )
        assert all(ranks[i] == 0 for i in fronts[0]), (
            "Front 0 solutions should have rank 0"
        )

        # Verify Front 1
        assert len(fronts[1]) == 2, (
            f"Front 1 should have 2 solutions, got {len(fronts[1])}"
        )
        assert all(ranks[i] == 1 for i in fronts[1]), (
            "Front 1 solutions should have rank 1"
        )

        # Verify Front 2
        assert len(fronts[2]) == 1, (
            f"Front 2 should have 1 solution, got {len(fronts[2])}"
        )
        assert ranks[5] == 2, "Last solution should be in Front 2"

        print(f"✅ Sorting correctly identified {len(fronts)} fronts!")
        print(f"   Front 0: {len(fronts[0])} solutions (non-dominated)")
        print(f"   Front 1: {len(fronts[1])} solutions")
        print(f"   Front 2: {len(fronts[2])} solutions")

    except NotImplementedError as e:
        print(f"❌ Not implemented: {str(e)}")
        print("   👉 Complete the TODOs in fast_nondominated_sort()")
        print("   See Exercise 2b in challenge2.md for guidance")
        exit(1)
    except AssertionError as e:
        print(f"❌ Assertion failed: {str(e)}")
        print("   👉 Check your sorting logic")
        exit(1)
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
        exit(1)

    # Optional tests for enhanced diversity (crowding distance)
    print("\n" + "-" * 60)
    print("Testing Enhanced Diversity (Crowding Distance)")
    print("-" * 60)

    # Test 3: Crowding Distance (OPTIONAL)
    print("\n[3/5] Testing Crowding Distance...")
    try:
        nsga = NSGAII(population_size=10, n_opt_params=5)

        # Create a simple front with known distances
        # Points: [1,1], [2,2], [3,3], [4,4], [5,5] (diagonal line)
        test_fitness = np.array(
            [
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5],
            ]
        )
        front_indices = [0, 1, 2, 3, 4]  # All in same front

        distances = nsga.compute_crowding_distance(test_fitness, front_indices)

        # Boundary solutions should have infinite distance
        assert distances[0] == np.inf, "First solution should have infinite distance"
        assert distances[4] == np.inf, "Last solution should have infinite distance"

        # Interior solutions should have finite positive distance
        assert distances[1] > 0 and np.isfinite(distances[1]), (
            "Interior solution should have finite positive distance"
        )
        assert distances[2] > 0 and np.isfinite(distances[2]), (
            "Interior solution should have finite positive distance"
        )
        assert distances[3] > 0 and np.isfinite(distances[3]), (
            "Interior solution should have finite positive distance"
        )

        print("✅ Crowding distance works correctly!")
        print(f"   Boundary distances: {distances[0]}, {distances[4]} (should be ∞)")
        print(
            f"   Interior distances: {distances[1]:.3f}, {distances[2]:.3f}, {distances[3]:.3f}"
        )

    except NotImplementedError:
        print("⏭️  Skipped (not implemented)")
        print("   This is optional. See challenge2.md for details if interested.")
    except AssertionError as e:
        print(f"⚠️  Implementation issue: {str(e)}")
    except Exception as e:
        print(f"⚠️  Error: {type(e).__name__}: {str(e)}")

    # Test 4: Crowding Operator (OPTIONAL)
    print("\n[4/5] Testing Crowding Operator...")
    try:
        nsga = NSGAII(population_size=10, n_opt_params=5)

        ranks = [0, 0, 1, 1]
        crowding_dists = np.array([2.0, 3.0, 5.0, 1.0])

        # Test rank preference (lower rank wins)
        winner = nsga.crowding_operator(0, 2, ranks, crowding_dists)
        assert winner == 0, "Individual with rank 0 should beat rank 1"

        # Test crowding distance preference (same rank)
        winner = nsga.crowding_operator(0, 1, ranks, crowding_dists)
        assert winner == 1, "Individual with crowding distance 3.0 should beat 2.0"

        print("✅ Crowding operator works correctly!")
        print("   ✓ Prefers lower rank (better front)")
        print("   ✓ Prefers larger crowding distance (more diverse)")

    except NotImplementedError:
        print("⏭️  Skipped (not implemented)")
        print("   This is optional. See challenge2.md for details if interested.")
    except AssertionError as e:
        print(f"⚠️  Implementation issue: {str(e)}")
    except Exception as e:
        print(f"⚠️  Error: {type(e).__name__}: {str(e)}")

    # Test 5: Enhanced Parent Selection (OPTIONAL)
    print("\n[5/5] Testing Enhanced Parent Selection...")

    try:
        nsga = NSGAII(population_size=10, n_opt_params=5, n_parents=5)

        # Create a small population
        test_population = np.random.uniform(-1, 1, (10, 5))
        test_fitness = np.random.uniform(0, 10, (10, 2))

        # Select parents - this works with or without crowding distance
        parents, parent_fitness = nsga.sort_and_select_parents(
            test_population, test_fitness, n_parents=5
        )

        assert parents.shape == (5, 5), (
            f"Should select 5 parents with 5 params, got shape {parents.shape}"
        )
        assert parent_fitness.shape == (5, 2), (
            f"Parent fitness should be (5, 2), got {parent_fitness.shape}"
        )

        print("✅ Parent selection works!")
        print(
            f"   Selected {len(parents)} parents from population of {len(test_population)}"
        )
        print("   Note: This uses rank-based selection (crowding distance is optional)")

    except AssertionError as e:
        print(f"⚠️  Implementation issue: {str(e)}")
    except Exception as e:
        print(f"⚠️  Error: {type(e).__name__}: {str(e)}")

    print("\n" + "=" * 60)
    print("🎉 NSGA-II tests passed!")
    print("=" * 60)
    print("\nYour implementation is ready to run multi-objective evolution.")
    print("Uncomment the evolution code below to start training.\n")


def inspect_ant_multi_world():
    """Test the AntMultiWorld environment."""
    world = AntMultiWorld(controller_cls=NeuralNetworkController)
    print(f"Observation space: {world.obs_size}")
    print(f"Action space: {world.action_size}")
    print(f"Controller parameters: {world.n_params}")

    # Test evaluation of a random individual
    random_genotype = np.random.uniform(-1, 1, world.n_params)
    fitness = world.evaluate_individual(random_genotype)
    print(f"Fitness of random individual: {fitness}")


def run_evolution_nsga(
    num_generations: int,
    population_size: int,
    ckpt_interval: int,
    checkpoint_path: Optional[str] = None,
    random_seed: int = 42,
) -> None:
    """Run evolutionary optimization for robot controller."""
    np.random.seed(random_seed)

    # Create world for evaluation
    world = AntMultiWorld(controller_cls=NeuralNetworkController)

    # Setup checkpoint directory
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if checkpoint_path is None:
        checkpoint_path = f"results/{dt_str}_nsga_ckpts"
    else:
        # If path is relative or absolute, just add prefix
        checkpoint_path = str(Path(checkpoint_path).parent / f"{dt_str}_{Path(checkpoint_path).name}")

    ckpt_dir = Path(checkpoint_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Create evolutionary algorithm
    num_params = world.n_params
    nsga = NSGAII(
        population_size=population_size,
        n_opt_params=num_params,
        n_parents=population_size,
        bounds=(-1, 1),
        mutation_prob=0.3,
        crossover_prob=0.5,
        output_dir=ckpt_dir,
    )

    # Evolution loop
    fitness_history = []
    best_overall_fitness = -np.inf
    best_overall_individual = None

    # Print training header
    print("\n" + "=" * 80)
    print(f"{'MULTI-OBJECTIVE EVOLUTION - NSGA-II':^80}")
    print("=" * 80)
    print(
        f"Population: {population_size} | Generations: {num_generations} | Parents: {nsga.n_parents}"
    )
    print(f"Objective 1: Flat Terrain Speed | Objective 2: Ice Terrain Speed")
    print("=" * 80 + "\n")

    for generation in range(num_generations):
        # Ask EA for new population
        population = nsga.ask()
        multi_fitness = np.empty((len(population), 2))

        for i, individual in enumerate(population):
            multi_fitness[i] = world.evaluate_individual(individual)

        # Tell EA the results
        save_checkpoint = (generation % ckpt_interval == 0) or (generation == num_generations - 1)
        nsga.tell(population, multi_fitness, save_checkpoint=save_checkpoint)

        # Track best individual (based on first objective for overall best tracking)
        gen_best_idx = np.argmax(multi_fitness[:, 0])
        gen_best_fitness = multi_fitness[gen_best_idx, 0]
        if gen_best_fitness > best_overall_fitness:
            best_overall_fitness = gen_best_fitness
            best_overall_individual = population[gen_best_idx].copy()

        # Logging metrics - show both objectives
        fitness_history.append(multi_fitness)
        mean_fitness_obj1 = np.mean(multi_fitness[:, 0])
        mean_fitness_obj2 = np.mean(multi_fitness[:, 1])
        best_obj1 = np.max(multi_fitness[:, 0])
        best_obj2 = np.max(multi_fitness[:, 1])

        # Progress bar
        progress = (generation + 1) / num_generations
        bar_length = 50
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)

        # Print with better formatting
        print(
            f"Gen {generation + 1:4d}/{num_generations} [{bar}] {progress * 100:5.1f}%"
        )
        print(
            f"     Best:     Objective 1={best_obj1:7.2f}  Objective 2={best_obj2:7.2f}"
        )
        print(
            f"     Mean:     Objective 1={mean_fitness_obj1:7.2f}  Objective 2={mean_fitness_obj2:7.2f}"
        )
        print()


def replay_checkpoint(checkpoint_path: str):
    np.random.seed(31)

    population = np.load(f"{checkpoint_path}/x.npy")

    world = AntMultiWorld(controller_cls=NeuralNetworkController)

    multi_fitness = np.empty((len(population), 2))

    for i, individual in enumerate(population):
        multi_fitness[i] = world.evaluate_individual(individual)

    best_flat_idx = np.argmax(multi_fitness[:, 0])
    best_ice_idx = np.argmax(multi_fitness[:, 1])

    print(
        f"Best Flat Terrain Individual: {best_flat_idx} with fitness {multi_fitness[best_flat_idx]}"
    )
    print(
        f"Best Ice Terrain Individual: {best_ice_idx} with fitness {multi_fitness[best_ice_idx]}"
    )

    plt.scatter(multi_fitness[:, 0], multi_fitness[:, 1], alpha=0.5)
    plt.xlabel("Fitness Objective 1")
    plt.ylabel("Fitness Objective 2")
    plt.title("Multi-Objective Fitness Scatter Plot")
    plt.savefig("fitness_scatter.png")

    n_evals = 5
    for idx_eval in range(n_evals):
        ant_ice_world = AntFlatWorld()
        ant_ice_world.generate_best_individual_video(
            env=ant_ice_world.create_env(
                robot_path="ant_ice_terrain.xml", width=800, height=608
            ),
            video_name=f"best_ice_individual_{idx_eval}.mp4",
            controller=ant_ice_world.geno2pheno(population[best_ice_idx]),
        )

        ant_flat_world = AntFlatWorld()
        ant_flat_world.generate_best_individual_video(
            env=ant_flat_world.create_env(
                robot_path="ant_flat_terrain.xml", width=800, height=608
            ),
            video_name=f"best_flat_individual_{idx_eval}.mp4",
            controller=ant_flat_world.geno2pheno(population[best_flat_idx]),
        )
        print(f"Generated videos for iteration {idx_eval + 1}/{n_evals}")


def plot_pareto_fronts_from_checkpoint(checkpoint_dir: str):
    """
    Loads fitness data from a checkpoint directory and plots Pareto fronts using NSGA-II sorting.
    """
    # Load all generations' fitness data
    fitness_path = f"{checkpoint_dir}/full_f.npy"
    try:
        all_fitness = np.load(fitness_path)
    except Exception as e:
        print(f"Could not load fitness data from {fitness_path}: {e}")
        return

    # If only 2D objectives, shape is (generations, pop, 2)
    if all_fitness.ndim == 3:
        # Plot only the last generation by default
        fitness = all_fitness[-1]
    else:
        fitness = all_fitness

    nsga = NSGAII(
        population_size=fitness.shape[0], n_opt_params=560
    )  # n_opt_params is a dummy here
    fronts, _ = nsga.fast_nondominated_sort(fitness)

    plt.figure(figsize=(8, 6))
    n_fronts = len(fronts)
    colors = plt.cm.viridis(np.linspace(0, 1, n_fronts))
    for i, front in enumerate(fronts[:n_fronts]):
        front_fitness = fitness[front]
        plt.scatter(
            front_fitness[:, 0],
            front_fitness[:, 1],
            label=f"Front {i}",
            color=colors[i],
            alpha=0.7,
        )

    plt.xlabel("Fitness Objective 1")
    plt.ylabel("Fitness Objective 2")
    plt.title("Pareto Fronts (Last Generation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pareto_fronts.png")
    plt.show()
    print("Pareto fronts plotted and saved as pareto_fronts.png")


if __name__ == "__main__":
    # Run unit tests first
    test_exercise_implementation()

    # Uncomment to run full NSGA-II evolution:
    run_evolution_nsga(
        num_generations=100,
        population_size=10,
        ckpt_interval=5,
        checkpoint_path=None,
        random_seed=42,
    )

    # Uncomment to replay your checkpoint
    # replay_checkpoint(
    #     checkpoint_path="./results/nsga_multi_terrain_ckpt/99"
    # )

    # Uncomment to plot Pareto fronts from checkpoint
    # plot_pareto_fronts_from_checkpoint(
    #     checkpoint_dir="./results/nsga_multi_terrain_ckpt/99"
    # )

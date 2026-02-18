import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from evorob.algorithms.ea_api_sol import EvoAlgAPI
from evorob.world.ant_world import AntFlatWorld
from evorob.world.robot.controllers.sinoid_sol import OscillatoryController


def test_exercise_implementation():
    print("\n" + "=" * 60)
    print("EXERCISE 1b: Testing Oscillatory Controller")
    print("=" * 60)

    # Test 1: Oscillatory Controller
    print("\n[1/2] Testing Oscillatory Controller...")
    try:
        controller = OscillatoryController(output_size=8)
        assert controller.n_params == 24, (
            f"Should have 24 params (3*8), got {controller.n_params}"
        )

        test_obs = np.random.randn(27)
        actions = controller.get_action(test_obs)
        assert actions.shape == (8,), (
            f"Action shape should be (8,), got {actions.shape}"
        )
        assert np.all(actions >= -1.0) and np.all(actions <= 1.0), (
            "Actions outside [-1, 1]"
        )

        # Test batched observations
        batched_obs = np.random.randn(4, 27)
        batched_actions = controller.get_action(batched_obs)
        assert batched_actions.shape == (4, 8), (
            f"Batched actions should be (4, 8), got {batched_actions.shape}"
        )

        # Test parameter setting
        test_weights = np.random.uniform(-1, 1, controller.n_params)
        controller.set_weights(test_weights)
        actions_after = controller.get_action(test_obs)
        assert actions_after.shape == (8,), "Actions shape changed after set_weights"

        print("✅ Oscillatory Controller works correctly!")
    except NotImplementedError as e:
        print(f"❌ Not implemented: {str(e)}")
        print(
            "   👉 Implement __init__(), get_action(), set_weights(), get_num_params() in sinoid.py"
        )
        exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        exit(1)

    # Test 2: Evolutionary Algorithm
    print("\n[2/2] Testing Evolutionary Algorithm API...")
    try:
        ea = EvoAlgAPI(n_params=24, population_size=20, sigma=0.5)
        population = ea.ask()
        assert population.shape == (20, 24), (
            f"Population shape should be (20, 24), got {population.shape}"
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
    print("\nUncomment the lines below to start evolutionary training.")


""" 
    Controller optimisation: Ant flat terrain
"""


def run_evolution_oscillatory_controller(
    num_generations: int,
    population_size: int,
    ckpt_interval: int,
    checkpoint_path: Optional[str] = None,
    run_evaluation: bool = True,
    random_seed: int = 42,
) -> None:
    """Run evolutionary optimization for robot controller."""
    np.random.seed(random_seed)

    # Create world for evaluation
    world = AntFlatWorld()
    world.controller = OscillatoryController(output_size=world.action_size)
    world.n_params = world.controller.n_params

    # Timestamped checkpoint directory
    dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if checkpoint_path is None:
        checkpoint_path = f"results/{dt_str}_oscillatory_controller_ckpts"
    else:
        # If path is relative or absolute, just add prefix
        checkpoint_path = str(Path(checkpoint_path).parent / f"{dt_str}_{Path(checkpoint_path).name}")

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
        save_checkpoint = (generation % ckpt_interval == 0) or (generation == num_generations - 1)
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
    if checkpoint_path:
        print(f"Checkpoints saved to {ckpt_dir}")

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
                    print(f"Trial {trial_count} reward: {float(trial_reward):.2f}")
                    trial_reward = 0.0
                    obs, _ = evaluation_env.reset()

        except KeyboardInterrupt:
            print(f"\n\nEvaluation stopped by user after {trial_count} trials.")
        finally:
            evaluation_env.close()


if __name__ == "__main__":
    test_exercise_implementation()

    # Uncomment to run full evolution:
    run_evolution_oscillatory_controller(
        num_generations=100,
        population_size=10, 
        ckpt_interval=5,
        checkpoint_path=None,
        run_evaluation=True,
        random_seed=42,
    )

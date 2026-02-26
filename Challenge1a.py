import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from evorob.algorithms.ea_api import EvoAlgAPI
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

        # Test _get_rew(): should return (reward, reward_info_dict) with three components
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


def run_evolution_neural_controller(
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
    world = AntFlatWorld(controller_cls=NeuralNetworkController)

    # Timestamped checkpoint directory
    dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if checkpoint_path is None:
        checkpoint_path = f"results/{dt_str}_neural_controller_ckpts"
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
    run_evolution_neural_controller(
        num_generations=100,
        population_size=10,
        ckpt_interval=5,
        checkpoint_path=None,
        run_evaluation=True,
        random_seed=42,
    )

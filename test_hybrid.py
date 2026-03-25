import datetime
import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from evorob.algorithms.ea_api import EvoAlgAPI
from evorob.world.envs.ant_flat import AntFlatEnvironment
from evorob.world.robot.controllers.mlp import NeuralNetworkController
from evorob.world.robot.controllers.sinoid import PhaseOscillatorController
from evorob.world.robot.controllers.hybrid import PhaseHybridResidualController


PPO_PATH = "results/ppo_ckpts/ppo_ant_42000000_steps.zip"
STATS_PATH = "results/ppo_ckpts/ppo_ant_vecnormalize_42000000_steps.pkl"


def evaluate_hybrid_once(vec_env, hybrid_ctrl, genotype, rollout_steps=1000):
    hybrid_ctrl.set_weights(genotype)
    obs = vec_env.reset()
    hybrid_ctrl.reset_controller()

    total_reward = 0.0
    for _ in range(rollout_steps):
        action = hybrid_ctrl.get_action(obs)
        obs, reward, done, _ = vec_env.step(action)
        total_reward += float(reward[0])
        if done[0]:
            break
    return total_reward


def format_vec(vec):
    return "[" + ", ".join(f"{float(v):.2f}" for v in vec) + "]"


def run_hybrid_evolution(
    num_generations: int = 200,
    population_size: int = 64,
    ckpt_interval: int = 10,
    sigma: float = 0.05,
    residual_scale: float = 0.05,
    random_seed: int = 42,
):
    np.random.seed(random_seed)

    # ------------------------------------------------------------
    # 1) Environment
    # ------------------------------------------------------------
    def make_env():
        return AntFlatEnvironment(
            render_mode=None,
            robot_path="/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/evorob/world/envs/assets/ant_ice_terrain.xml"
        )

    vec_env = DummyVecEnv([make_env])

    if not os.path.exists(STATS_PATH):
        print("❌ CRITICAL ERROR: VecNormalize stats file not found.")
        print(f"   Expected: {STATS_PATH}")
        return

    vec_env = VecNormalize.load(STATS_PATH, vec_env)
    vec_env.training = False
    vec_env.norm_obs = False
    vec_env.norm_reward = False

    print("✅ Loaded VecNormalize stats.")
    print("✅ Observation normalization enabled.")
    print("✅ Reward normalization disabled for raw fitness.")

    # ------------------------------------------------------------
    # 2) Frozen PPO controller
    # ------------------------------------------------------------
    if not os.path.exists(PPO_PATH):
        print("❌ PPO checkpoint not found.")
        print(f"   Expected: {PPO_PATH}")
        return

    model = PPO.load(PPO_PATH, device="cpu")

    nn_ctrl = NeuralNetworkController(
        input_size=27,
        output_size=8,
        hidden_size=[256, 256],
    )
    nn_ctrl.load_from_ppo_model(model)

    # ------------------------------------------------------------
    # 3) Residual controller + phase oscillator + hybrid
    # ------------------------------------------------------------
    obs_dim = 27
    action_dim = 8
    phase_feat_dim = 2 * action_dim  # sin + cos for each joint

    residual_ctrl = NeuralNetworkController(
        input_size=obs_dim + phase_feat_dim,
        output_size=action_dim,
        hidden_size=[16],   # keep small at first
    )

    # initialize residual near zero so controller starts as pure PPO
    residual_x0 = np.zeros(residual_ctrl.get_num_params(), dtype=np.float32)
    residual_ctrl.set_weights(residual_x0)

    phase_ctrl = PhaseOscillatorController(
        output_size=action_dim,
        dt=0.01,
        default_frequency=1.0,
    )

    hybrid_ctrl = PhaseHybridResidualController(
        ppo_controller=nn_ctrl,
        residual_controller=residual_ctrl,
        phase_controller=phase_ctrl,
        residual_scale=residual_scale,
        action_dim=action_dim,
    )

    # ------------------------------------------------------------
    # 4) Initial genotype
    # ------------------------------------------------------------
    n_res = residual_ctrl.get_num_params()
    n_phase = phase_ctrl.get_num_params()
    n_params = n_res + n_phase

    x0 = np.zeros(n_params, dtype=np.float32)

    # phase oscillator initial frequencies
    x0[n_res:n_res + action_dim] = 1.0

    # phase offsets
    x0[n_res + action_dim:n_res + 2 * action_dim] = 0.0

    # ------------------------------------------------------------
    # 5) Output directory
    # ------------------------------------------------------------
    dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(f"results/{dt_str}_phase_hybrid_residual_ckpts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 6) CMA-ES
    # ------------------------------------------------------------
    ea = EvoAlgAPI(
        n_params=n_params,
        population_size=population_size,
        sigma=sigma,
        x0=x0,
        output_dir=ckpt_dir,
    )

    # ------------------------------------------------------------
    # 7) Evaluate exact seed once
    # ------------------------------------------------------------
    seed_fitness = evaluate_hybrid_once(vec_env, hybrid_ctrl, x0)
    hybrid_ctrl.set_weights(x0)
    phase_info = hybrid_ctrl.get_phase_info()

    print("\n" + "=" * 72)
    print("PHASE HYBRID EVOLUTION: FROZEN PPO + TRAINABLE RESIDUAL + PHASE FEATURES")
    print("=" * 72)
    print(f"Residual scale      : {residual_scale}")
    print(f"Residual params     : {n_res}")
    print(f"Phase params        : {n_phase}")
    print(f"Total params        : {n_params}")
    print(f"Initial seed fitness: {seed_fitness:.2f}")
    print(f"Initial frequencies : {format_vec(phase_info['frequencies'])}")
    print(f"Initial phases      : {format_vec(phase_info['phases'])}")
    print("=" * 72)

    # ------------------------------------------------------------
    # 8) Evolution loop
    # ------------------------------------------------------------
    for generation in range(num_generations):
        population = ea.ask()

        if generation == 0:
            population[0] = x0.copy()

        fitness = np.empty(len(population), dtype=np.float32)

        for i, individual in enumerate(population):
            fitness[i] = evaluate_hybrid_once(vec_env, hybrid_ctrl, individual)

        prev_best = float(ea.f_best_so_far)

        save_checkpoint = (
            (generation % ckpt_interval == 0)
            or (generation == num_generations - 1)
        )

        ea.tell(population, fitness, save_checkpoint=save_checkpoint)

        gen_best_idx = int(np.argmax(fitness))
        gen_best_fitness = float(fitness[gen_best_idx])
        mean_fitness = float(np.mean(fitness))
        median_fitness = float(np.median(fitness))
        best_individual = population[gen_best_idx].copy()

        hybrid_ctrl.set_weights(best_individual)
        phase_info = hybrid_ctrl.get_phase_info()

        print(
            f"Generation {generation + 1}/{num_generations}: "
            f"Best={gen_best_fitness:.2f}, "
            f"Mean={mean_fitness:.2f}, "
            f"Median={median_fitness:.2f}, "
            f"Overall Best={ea.f_best_so_far:.2f}"
            + (f" | x0 residual fitness={seed_fitness:.2f}" if generation == 0 else "")
        )
        print(f"  Frequencies: {format_vec(phase_info['frequencies'])}")
        print(f"  Phases     : {format_vec(phase_info['phases'])}")

        if float(ea.f_best_so_far) > prev_best:
            np.save(ckpt_dir / "x_best_hybrid_running.npy", ea.x_best_so_far)

    np.save(ckpt_dir / "x0_used.npy", x0)
    if ea.x_best_so_far is not None:
        np.save(ckpt_dir / "x_best_final.npy", ea.x_best_so_far)

    print("\n" + "=" * 72)
    print(f"Evolution complete! Best fitness: {ea.f_best_so_far:.2f}")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print("=" * 72)

    vec_env.close()


if __name__ == "__main__":
    run_hybrid_evolution(
        num_generations=100,
        population_size=100,
        ckpt_interval=5,
        sigma=0.5,
        residual_scale=0.15,
        random_seed=42,
    )
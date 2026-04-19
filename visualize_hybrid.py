import os
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from evorob.world.envs.ant_flat import AntFlatEnvironment
from evorob.world.robot.controllers.mlp import NeuralNetworkController
from evorob.world.robot.controllers.sinoid import PhaseOscillatorController
from evorob.world.robot.controllers.hybrid import PhaseHybridResidualController


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PPO_PATH = "results/ppo_ckpts/ppo_ant_10000000_steps.zip"
STATS_PATH = "results/ppo_ckpts/ppo_ant_vecnormalize_10000000_steps.pkl"
CHECKPOINT_DIR = "/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/results/20260319_101259_phase_hybrid_residual_ckpts_best_sofar ice/"


def find_best_genotype(checkpoint_dir: str) -> str:
    """
    Try the most likely saved files in order.
    """
    candidates = [
        os.path.join(checkpoint_dir, "x_best_final.npy"),
        os.path.join(checkpoint_dir, "x_best_hybrid_running.npy"),
        os.path.join(checkpoint_dir, "x0_used.npy"),
        os.path.join(checkpoint_dir, "full_x.npy"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Could not find a genotype in {checkpoint_dir}. "
        f"Tried: {candidates}"
    )


def format_vec(vec):
    return "[" + ", ".join(f"{float(v):.2f}" for v in vec) + "]"


def make_env(render_mode=None):
    return AntFlatEnvironment(
        render_mode=render_mode,
        robot_path="/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/evorob/world/envs/assets/ant_flat_terrain.xml",
    )


def build_controller():
    # ------------------------------------------------------------
    # 1) Load PPO
    # ------------------------------------------------------------
    if not os.path.exists(PPO_PATH):
        raise FileNotFoundError(f"PPO checkpoint not found: {PPO_PATH}")
    model = PPO.load(PPO_PATH, device="cpu")

    # ------------------------------------------------------------
    # 2) Frozen PPO controller
    # ------------------------------------------------------------
    ppo_ctrl = NeuralNetworkController(
        input_size=27,
        output_size=8,
        hidden_size=[256, 256],
    )
    ppo_ctrl.load_from_ppo_model(model)

    # ------------------------------------------------------------
    # 3) Residual controller
    # input = obs + phase features = 27 + 16
    # ------------------------------------------------------------
    residual_ctrl = NeuralNetworkController(
        input_size=27 + 16,
        output_size=8,
        hidden_size=[16],
    )

    # initialize residual near zero
    residual_zero = np.zeros(residual_ctrl.get_num_params(), dtype=np.float32)
    residual_ctrl.set_weights(residual_zero)

    # ------------------------------------------------------------
    # 4) Phase oscillator
    # ------------------------------------------------------------
    phase_ctrl = PhaseOscillatorController(
        output_size=8,
        dt=0.01,
        default_frequency=1.0,
    )

    # ------------------------------------------------------------
    # 5) Hybrid controller
    # ------------------------------------------------------------
    hybrid_ctrl = PhaseHybridResidualController(
        ppo_controller=ppo_ctrl,
        residual_controller=residual_ctrl,
        phase_controller=phase_ctrl,
        residual_scale=0.05,
        action_dim=8,
    )

    return hybrid_ctrl


def play_hybrid_controller(
    checkpoint_dir: str,
    max_steps: int = 1000,
    n_episodes: int = 5,
):
    # ------------------------------------------------------------
    # Build vec env with rendering
    # ------------------------------------------------------------
    vec_env = DummyVecEnv([lambda: make_env(render_mode="human")])

    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError(f"VecNormalize stats not found: {STATS_PATH}")

    vec_env = VecNormalize.load(STATS_PATH, vec_env)
    vec_env.training = False
    vec_env.norm_obs = True
    vec_env.norm_reward = False

    # ------------------------------------------------------------
    # Build controller
    # ------------------------------------------------------------
    hybrid_ctrl = build_controller()

    # ------------------------------------------------------------
    # Load genotype
    # ------------------------------------------------------------
    genotype_path = find_best_genotype(checkpoint_dir)
    genotype = np.load(genotype_path).astype(np.float32).ravel()

    expected_params = hybrid_ctrl.get_num_params()
    if genotype.shape[0] != expected_params:
        raise ValueError(
            f"Expected genotype of size {expected_params} for phase hybrid, "
            f"got {genotype.shape[0]}"
        )

    hybrid_ctrl.set_weights(genotype)
    phase_info = hybrid_ctrl.get_phase_info()

    print("=" * 72)
    print("PLAYING PHASE HYBRID CONTROLLER")
    print("=" * 72)
    print(f"Loaded genotype: {genotype_path}")
    print(f"Total params   : {expected_params}")
    print(f"Frequencies    : {format_vec(phase_info['frequencies'])}")
    print(f"Phases         : {format_vec(phase_info['phases'])}")
    print("=" * 72)

    try:
        for ep in range(n_episodes):
            obs = vec_env.reset()
            hybrid_ctrl.reset_controller()
            total_reward = 0.0

            for step in range(max_steps):
                action = hybrid_ctrl.get_action(obs)
                obs, reward, done, _ = vec_env.step(action)
                total_reward += float(reward[0])

                if done[0]:
                    break

            print(f"Episode {ep + 1}/{n_episodes}: reward = {total_reward:.2f}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        vec_env.close()


if __name__ == "__main__":
    play_hybrid_controller(
        checkpoint_dir=CHECKPOINT_DIR,
        max_steps=1000,
        n_episodes=5,
    )
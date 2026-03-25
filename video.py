import os
import imageio
import numpy as np

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
CHECKPOINT_DIR = "results/20260318_185115_phase_hybrid_residual_ckpts_best so far ice"
OUTPUT_VIDEO = "video_output/hybrid_evaluation_video.mp4"


def find_best_genotype(checkpoint_dir: str) -> str:
    candidates = [
        os.path.join(checkpoint_dir, "x_best_final.npy"),
        os.path.join(checkpoint_dir, "x_best_hybrid_running.npy"),
        os.path.join(checkpoint_dir, "x0_used.npy"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Could not find a genotype in {checkpoint_dir}. Tried: {candidates}"
    )


def make_env(render_mode=None):
    return AntFlatEnvironment(
        render_mode=render_mode,
        robot_path="/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/evorob/world/envs/assets/ant_ice_terrain.xml",
    )


def build_controller():
    model = PPO.load(PPO_PATH, device="cpu")

    ppo_ctrl = NeuralNetworkController(
        input_size=27,
        output_size=8,
        hidden_size=[256, 256],
    )
    ppo_ctrl.load_from_ppo_model(model)

    residual_ctrl = NeuralNetworkController(
        input_size=27 + 16,   # obs + phase features
        output_size=8,
        hidden_size=[16],
    )
    residual_zero = np.zeros(residual_ctrl.get_num_params(), dtype=np.float32)
    residual_ctrl.set_weights(residual_zero)

    phase_ctrl = PhaseOscillatorController(
        output_size=8,
        dt=0.01,
        default_frequency=1.0,
    )

    hybrid_ctrl = PhaseHybridResidualController(
        ppo_controller=ppo_ctrl,
        residual_controller=residual_ctrl,
        phase_controller=phase_ctrl,
      
    )

    return hybrid_ctrl


def save_hybrid_video(
    checkpoint_dir: str,
    output_video: str,
    max_steps: int = 1000,
    seed: int = 0,
):
    # ------------------------------------------------------------
    # Build vec env with rgb rendering
    # ------------------------------------------------------------
    vec_env = DummyVecEnv([lambda: make_env(render_mode="rgb_array")])

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
            f"Expected genotype of size {expected_params}, got {genotype.shape[0]}"
        )

    hybrid_ctrl.set_weights(genotype)

    # ------------------------------------------------------------
    # Reset env and record frames
    # ------------------------------------------------------------
    obs = vec_env.reset()
    try:
        vec_env.env_method("reset", seed=seed)
        obs = vec_env.reset()
    except Exception:
        pass

    hybrid_ctrl.reset_controller()
    frames = []
    total_reward = 0.0

    print("Recording video...")

    for _ in range(max_steps):
        # render underlying env frame
        frame = vec_env.venv.envs[0].render()
        frames.append(frame)

        action = hybrid_ctrl.get_action(obs)
        obs, reward, done, _ = vec_env.step(action)
        total_reward += float(reward[0])

        if done[0]:
            break

    vec_env.close()

    # ------------------------------------------------------------
    # Save video
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    imageio.mimwrite(output_video, frames, fps=20)

    print(f"Video saved to: {output_video}")
    print(f"Episode reward: {total_reward:.2f}")


if __name__ == "__main__":
    save_hybrid_video(
        checkpoint_dir=CHECKPOINT_DIR,
        output_video=OUTPUT_VIDEO,
        max_steps=50,
        seed=0,
    )
import os
import imageio
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from evorob.utils.filesys import get_last_checkpoint_dir
from evorob.world.envs.ant_flat import AntFlatEnvironment
from evorob.world.robot.controllers.mlp import NeuralNetworkController
from evorob.world.robot.controllers.sinoid import PhaseOscillatorController
from evorob.world.robot.controllers.hybrid import PhaseHybridResidualController


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PPO_PATH = "results/ppo_ckpts/ppo_ant_10000000_steps.zip"
STATS_PATH = "results/ppo_ckpts/ppo_ant_vecnormalize_10000000_steps.pkl"
CHECKPOINT_DIR = "/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/results/20260319_101259_phase_hybrid_residual_ckpts_best_sofar ice"
OUTPUT_VIDEO = "video_output/hybrid_evaluation_video.mp4"


def find_best_genotype(checkpoint_dir: str) -> str:
    candidates = [
        os.path.join(checkpoint_dir, "80/x_best.npy"),
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
        robot_path="/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/evorob/world/envs/assets/ant_flat_terrain.xml",
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
    import gymnasium as gym
    # ------------------------------------------------------------
    # Build vec env with rgb rendering
    # ------------------------------------------------------------
    vec_env = DummyVecEnv([lambda: gym.make("Ant-v5", render_mode="rgb_array", include_cfrc_ext_in_observation=False)])

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
    rew_hist = []
    print("Recording video...")
    for i in range(256):
        frames = []
        total_reward = 0.0
        for _ in range(max_steps):
            # render underlying env frame
            frame = vec_env.venv.envs[0].render()
            frames.append(frame)

            action = hybrid_ctrl.get_action(obs)
            obs, reward, done, _ = vec_env.step(action)
            total_reward += float(reward[0])

            if done[0]:
                break

        rew_hist.append(total_reward)
        print(f"Episode {i + 1}: reward = {total_reward:.2f}  |  Last 10 ep rewards: {rew_hist[-10:]}")

    print(np.array(rew_hist).mean(), np.array(rew_hist).std())
    vec_env.close()

    # ------------------------------------------------------------
    # Save video
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    #imageio.mimwrite(output_video, frames, fps=20)

    print(f"Video saved to: {output_video}")
    print(f"Episode reward: {total_reward:.2f}")

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
    save_hybrid_video(
        checkpoint_dir=CHECKPOINT_DIR,
        output_video=OUTPUT_VIDEO,
        max_steps=1000,
        seed=0,
    )
    #evaluate_checkpoint(
    # checkpoint_dir=CHECKPOINT_DIR,
    # )
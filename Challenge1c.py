from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder, DummyVecEnv

from evorob.world.envs.ant_flat_sol import AntFlatEnvironment

""" 
    Controller optimisation: Ant flat terrain
"""

def run_reinforcement_learning(
    total_timesteps: int,
    num_envs: int,
    batch_size: int,
    checkpoint_path: Optional[str] = None,
    run_evaluation: bool = True,
    random_seed: int = 42,
) -> None:
    """Train a PPO agent using stable-baselines3."""
    env = make_vec_env(AntFlatEnvironment, n_envs=num_envs, vec_env_cls=DummyVecEnv)
    vec_env = VecNormalize(
        env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=2048,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=1e-4,
        clip_range=0.15,
        ent_coef=0.005,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
        seed=random_seed,
        device="cpu",
    )

    checkpoint_callback = None
    if checkpoint_path is not None:
        # Save checkpoint every 1M timesteps
        # save_freq is in env.step() calls, so divide by num_envs
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1_000_000 // num_envs, 1),
            save_path=checkpoint_path,
            name_prefix="ppo_ant",
            save_replay_buffer=False,
            save_vecnormalize=True,  # crucial
        )

    # Train the agent
    print(f"Training PPO for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    if checkpoint_path:
        log_dir = Path(checkpoint_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        model.save(log_dir / "model")
        stats_path = log_dir / "vec_normalize.pkl"
        vec_env.save(stats_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

    vec_env.close()

    if run_evaluation:
        replay_checkpoint(checkpoint_path=checkpoint_path)


def single_replay_checkpoint(
    checkpoint_path: Optional[str] = None,
    video_folder: str = ".",
    max_episode_steps: int = 1000,
) -> None:
    """Record a single replay of a checkpoint to video."""
    if checkpoint_path is None:
        print("Error: checkpoint_path is required to replay a checkpoint.")
        return

    checkpoint_path_obj = Path(checkpoint_path)

    # Check if checkpoint_path is a directory or a specific checkpoint file
    if checkpoint_path_obj.is_dir():
        model_path = checkpoint_path_obj / "model.zip"
        stats_path = checkpoint_path_obj / "vec_normalize.pkl"
    else:
        model_path = checkpoint_path_obj
        checkpoint_name = checkpoint_path_obj.stem
        vecnormalize_name = (
            checkpoint_name.replace("ppo_ant", "ppo_ant_vecnormalize") + ".pkl"
        )
        stats_path = checkpoint_path_obj.parent / vecnormalize_name

    if not model_path.exists() or not stats_path.exists():
        print(f"Error: Checkpoint files not found.")
        print(f"  Model path: {model_path} (exists: {model_path.exists()})")
        print(f"  Stats path: {stats_path} (exists: {stats_path.exists()})")
        return

    # Create environment with rgb_array mode for recording
    eval_env = make_vec_env(
        AntFlatEnvironment, n_envs=1, env_kwargs={"render_mode": "rgb_array"}
    )

    # Wrap with video recorder
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=video_folder,
        record_video_trigger=lambda x: x == 0,  # Record first episode
        video_length=max_episode_steps,
        name_prefix="ant_replay",
    )

    eval_env = VecNormalize.load(stats_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(model_path, env=eval_env, device="cpu")

    obs = eval_env.reset()
    episode_reward = 0.0
    episode_steps = 0

    print(f"Recording single episode (max {max_episode_steps} steps)...")
    done = False
    while not done and episode_steps < max_episode_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        episode_reward += reward[0]
        episode_steps += 1
        done = done[0]

    print(f"Episode completed: {episode_steps} steps, reward: {episode_reward:.2f}")
    eval_env.close()


def replay_checkpoint(checkpoint_path: str) -> None:
    """Load a checkpoint and run it in the environment."""
    checkpoint_path_obj = Path(checkpoint_path)

    # Check if checkpoint_path is a directory or a specific checkpoint file
    if checkpoint_path_obj.is_dir():
        # Use the final model/vec_normalize files
        model_path = checkpoint_path_obj / "model.zip"
        stats_path = checkpoint_path_obj / "vec_normalize.pkl"
    else:
        # checkpoint_path points to a specific checkpoint file
        model_path = checkpoint_path_obj
        # Derive the vecnormalize file from the checkpoint name
        # e.g., ppo_ant_1000000_steps.zip -> ppo_ant_vecnormalize_1000000_steps.pkl
        checkpoint_name = checkpoint_path_obj.stem  # removes .zip
        vecnormalize_name = (
            checkpoint_name.replace("ppo_ant", "ppo_ant_vecnormalize") + ".pkl"
        )
        stats_path = checkpoint_path_obj.parent / vecnormalize_name

    if not model_path.exists() or not stats_path.exists():
        print(f"Error: Checkpoint files not found.")
        print(f"  Model path: {model_path} (exists: {model_path.exists()})")
        print(f"  Stats path: {stats_path} (exists: {stats_path.exists()})")
        return

    eval_env = make_vec_env(
        AntFlatEnvironment, n_envs=1, env_kwargs={"render_mode": "human"}
    )
    eval_env = VecNormalize.load(stats_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(model_path, env=eval_env)

    obs = eval_env.reset()
    trial_reward = 0.0
    trial_count = 0
    episode_steps = 0
    max_episode_steps = 1000

    print("Press Ctrl+C to stop the replay...")
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            trial_reward += reward[0]
            episode_steps += 1

            # Reset if done or max episode steps reached
            if done[0] or episode_steps >= max_episode_steps:
                trial_count += 1
                print(
                    f"Trial {trial_count} reward: {trial_reward:.2f} (steps: {episode_steps})"
                )
                trial_reward = 0.0
                episode_steps = 0
                obs = eval_env.reset()

    except KeyboardInterrupt:
        print(f"\n\nReplay stopped by user after {trial_count} trials.")
    finally:
        eval_env.close()


if __name__ == "__main__":
    run_reinforcement_learning(
        total_timesteps=10_000, # recommended: 10_000_000 for good performance
        num_envs=16,
        batch_size=1024,
        run_evaluation=True,
        checkpoint_path="./results/ppo_ckpts"
    )

    # replay_checkpoint("./results/ppo_ckpts_10000_steps.zip")

    # single_replay_checkpoint(
    #     checkpoint_path="./results/ppo_ckpts_10000_steps.zip",
    #     video_folder=".",
    #     max_episode_steps=1000,
    # )

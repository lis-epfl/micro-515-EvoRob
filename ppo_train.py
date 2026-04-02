from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder, DummyVecEnv

from evorob.world.envs.ant_flat import AntFlatEnvironment

"""
Controller optimisation: Ant on both flat + ice terrains
"""


# ---------------------------------------------------------------------------
# Mixed-terrain environment
# ---------------------------------------------------------------------------

class AntTwoTerrainEnvironment(gym.Env):
    """
    Wrapper around two AntFlatEnvironment instances:
      - flat terrain
      - ice terrain

    Modes:
        - terrain_mode="random" : choose flat or ice at every episode reset
        - terrain_mode="flat"   : always flat
        - terrain_mode="ice"    : always ice
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    FLAT_ROBOT_PATH = "ant_flat_terrain.xml"
    ICE_ROBOT_PATH = "ant_ice_terrain.xml"

    def __init__(
        self,
        terrain_mode: str = "random",
        flat_prob: float = 0.5,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.terrain_mode = terrain_mode
        self.flat_prob = float(flat_prob)
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)

        self.flat_env = AntFlatEnvironment(
            robot_path=self.FLAT_ROBOT_PATH,
            render_mode=render_mode,
            **kwargs,
        )
        self.ice_env = AntFlatEnvironment(
            robot_path=self.ICE_ROBOT_PATH,
            render_mode=render_mode,
            **kwargs,
        )

        self.current_env = self.flat_env
        self.current_terrain = "flat"

        self.action_space = self.flat_env.action_space
        self.observation_space = self.flat_env.observation_space

    def _select_env(self):
        if self.terrain_mode == "flat":
            self.current_env = self.flat_env
            self.current_terrain = "flat"
        elif self.terrain_mode == "ice":
            self.current_env = self.ice_env
            self.current_terrain = "ice"
        elif self.terrain_mode == "random":
            if self._rng.random() < self.flat_prob:
                self.current_env = self.flat_env
                self.current_terrain = "flat"
            else:
                self.current_env = self.ice_env
                self.current_terrain = "ice"
        else:
            raise ValueError(
                f"Unknown terrain_mode='{self.terrain_mode}'. "
                f"Use 'random', 'flat', or 'ice'."
            )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._select_env()
        obs, info = self.current_env.reset(seed=seed, options=options)
        info = dict(info)
        info["terrain"] = self.current_terrain
        return obs, info

    def step(self, action):
        return self.current_env.step(action)

    def render(self):
        return self.current_env.render()

    def close(self):
        self.flat_env.close()
        self.ice_env.close()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_reinforcement_learning(
    total_timesteps: int,
    num_envs: int,
    batch_size: int,
    checkpoint_path: Optional[str] = None,
    run_evaluation: bool = True,
    random_seed: int = 42,
) -> None:
    """Train a PPO agent on both terrains using terrain randomization."""
    env = make_vec_env(
        AntTwoTerrainEnvironment,
        n_envs=num_envs,
        vec_env_cls=DummyVecEnv,
        seed=random_seed,
        env_kwargs={
            "terrain_mode": "random",
            "flat_prob": 0.5,
        },
    )

    vec_env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
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
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1_000_000 // num_envs, 1),
            save_path=checkpoint_path,
            name_prefix="ppo_ant_both",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )

    print(f"Training PPO for {total_timesteps} timesteps on flat + ice...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    if checkpoint_path:
        log_dir = Path(checkpoint_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        model.save(log_dir / "model")
        vec_env.save(log_dir / "vec_normalize.pkl")
        print(f"Model checkpoint saved to {checkpoint_path}")

    vec_env.close()

    if run_evaluation:
        print("\nEvaluating final model on flat terrain...")
        replay_checkpoint(checkpoint_path=checkpoint_path, terrain_mode="flat")

        print("\nEvaluating final model on ice terrain...")
        replay_checkpoint(checkpoint_path=checkpoint_path, terrain_mode="ice")


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def _resolve_model_and_stats(checkpoint_path: str, prefix: str = "ppo_ant_both"):
    checkpoint_path_obj = Path(checkpoint_path)

    if checkpoint_path_obj.is_dir():
        model_path = checkpoint_path_obj / "model.zip"
        stats_path = checkpoint_path_obj / "vec_normalize.pkl"
    else:
        model_path = checkpoint_path_obj
        checkpoint_name = checkpoint_path_obj.stem
        vecnormalize_name = checkpoint_name.replace(prefix, f"{prefix}_vecnormalize") + ".pkl"
        stats_path = checkpoint_path_obj.parent / vecnormalize_name

    return model_path, stats_path


def single_replay_checkpoint(
    checkpoint_path: Optional[str] = None,
    video_folder: str = ".",
    max_episode_steps: int = 1000,
    terrain_mode: str = "flat",
) -> None:
    """Record a single replay of a checkpoint to video on chosen terrain."""
    if checkpoint_path is None:
        print("Error: checkpoint_path is required to replay a checkpoint.")
        return

    model_path, stats_path = _resolve_model_and_stats(checkpoint_path, prefix="ppo_ant_both")

    if not model_path.exists() or not stats_path.exists():
        print("Error: Checkpoint files not found.")
        print(f"  Model path: {model_path} (exists: {model_path.exists()})")
        print(f"  Stats path: {stats_path} (exists: {stats_path.exists()})")
        return

    eval_env = make_vec_env(
        AntTwoTerrainEnvironment,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs={
            "render_mode": "rgb_array",
            "terrain_mode": terrain_mode,
        },
    )

    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=max_episode_steps,
        name_prefix=f"ant_replay_{terrain_mode}",
    )

    eval_env = VecNormalize.load(stats_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(model_path, env=eval_env, device="cpu")

    obs = eval_env.reset()
    episode_reward = 0.0
    episode_steps = 0

    print(f"Recording single {terrain_mode} episode (max {max_episode_steps} steps)...")
    done = False
    while not done and episode_steps < max_episode_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        episode_reward += reward[0]
        episode_steps += 1
        done = done[0]

    print(f"Episode completed: {episode_steps} steps, reward: {episode_reward:.2f}")
    eval_env.close()


def replay_checkpoint(checkpoint_path: str, terrain_mode: str = "flat") -> None:
    """Load a checkpoint and run it in the environment."""
    model_path, stats_path = _resolve_model_and_stats(checkpoint_path, prefix="ppo_ant_both")

    if not model_path.exists() or not stats_path.exists():
        print("Error: Checkpoint files not found.")
        print(f"  Model path: {model_path} (exists: {model_path.exists()})")
        print(f"  Stats path: {stats_path} (exists: {stats_path.exists()})")
        return

    eval_env = make_vec_env(
        AntTwoTerrainEnvironment,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs={
            "render_mode": "human",
            "terrain_mode": terrain_mode,
        },
    )

    eval_env = VecNormalize.load(stats_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(model_path, env=eval_env, device="cpu")

    obs = eval_env.reset()
    trial_reward = 0.0
    trial_count = 0
    episode_steps = 0
    max_episode_steps = 1000

    print(f"Press Ctrl+C to stop the replay on {terrain_mode} terrain...")
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            trial_reward += reward[0]
            episode_steps += 1

            if done[0] or episode_steps >= max_episode_steps:
                trial_count += 1
                print(
                    f"[{terrain_mode}] Trial {trial_count} reward: "
                    f"{trial_reward:.2f} (steps: {episode_steps})"
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
        total_timesteps=50_000_000,
        num_envs=16,
        batch_size=1024,
        run_evaluation=True,
        checkpoint_path="./results/ppo_ckpts_both",
        random_seed=42,
    )

    replay_checkpoint("./results/ppo_ckpts_both", terrain_mode="flat")
    replay_checkpoint("./results/ppo_ckpts_both", terrain_mode="ice")

    single_replay_checkpoint(
        checkpoint_path="./results/ppo_ckpts_both",
        video_folder=".",
        max_episode_steps=1000,
        terrain_mode="flat",
    )

    single_replay_checkpoint(
        checkpoint_path="./results/ppo_ckpts_both",
        video_folder=".",
        max_episode_steps=1000,
        terrain_mode="ice",
    )
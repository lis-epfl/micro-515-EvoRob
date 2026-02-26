import numpy as np
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnasium.wrappers import ClipAction, TimeLimit

from evorob.world.base import World
from evorob.world.envs.ant_flat import AntFlatEnvironment
from evorob.world.robot.controllers.base import Controller
from evorob.world.robot.controllers.mlp import NeuralNetworkController


class AntFlatWorld(World):
    """Wrapper for the Ant environment for evolutionary optimization."""

    def __init__(self, controller_cls: type[Controller] = NeuralNetworkController):
        self.env = self.create_env()
        self.dt = self.env.envs[0].unwrapped.dt
        self.action_size = self.env.action_space.shape[1]
        self.obs_size = self.env.observation_space.shape[1]
        self.controller = controller_cls(self.obs_size, self.action_size)
        self.n_params = self.controller.n_params
        self._eval_counter = 0  # Counter for seed generation

    def create_env(
        self,
        render_mode: str = "rgb_array",
        n_repeats: int = 1,
        max_episode_steps: int = 1000,
        **kwargs,
    ):
        """Create the Ant environment with proper wrappers."""
        self.n_repeats = n_repeats

        def make_env(param):
            def _init():
                env = AntFlatEnvironment(render_mode=render_mode, **kwargs)
                env = ClipAction(env)
                env = TimeLimit(env, max_episode_steps=max_episode_steps)
                return env

            return _init

        env_fns = [make_env(i) for i in range(n_repeats)]
        vec_env = SyncVectorEnv(env_fns)

        return vec_env

    def geno2pheno(self, genotype):
        """Convert genotype to phenotype (controller weights)."""
        self.controller.geno2pheno(genotype)
        return self.controller

    def evaluate_individual(self, genotype, trial_time: int = 20):
        """Evaluate a single individual (genotype) in the environment."""
        n_sim_steps = int(trial_time / self.dt)

        self.geno2pheno(genotype)

        # Generate unique seeds for each environment in this evaluation
        seeds = [self._eval_counter * self.n_repeats + i for i in range(self.n_repeats)]
        self._eval_counter += 1
        
        observations, _ = self.env.reset(seed=seeds)
        done_mask = np.zeros(self.n_repeats, dtype=bool)
        rewards_full = np.zeros((n_sim_steps, self.n_repeats))
        for step in range(n_sim_steps):
            action = self.controller.get_action(observations)
            observations, rewards, terminated, truncated, _ = self.env.step(action)
            rewards_full[step, ~done_mask] = rewards[~done_mask]

            done_mask = done_mask | terminated | truncated

            if np.all(done_mask):
                break

        final_rewards = np.sum(rewards_full, axis=0)
        return np.mean(final_rewards)
    
    def update_robot_xml(self, genotype: np.ndarray):
        """Update the robot's XML based on the genotype."""
        # For AntFlatWorld, we don't have body parameters.
        pass

    def close(self):
        """Explicitly close the environment."""
        if hasattr(self, "env") and self.env is not None:
            self.env.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
        return False

    def __del__(self):
        """Fallback cleanup if close() wasn't called explicitly."""
        self.close()

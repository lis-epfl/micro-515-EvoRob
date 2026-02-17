from abc import ABC, abstractmethod

import numpy as np
from gymnasium import Env
from gymnasium.vector import AsyncVectorEnv

from evorob.world.robot.controllers.base import Controller


class World(ABC):

    controller: Controller

    @abstractmethod
    def create_env(self, render_mode: str = 'rgb_array', **kwargs) -> AsyncVectorEnv|Env:
        return NotImplementedError

    @abstractmethod
    def evaluate_individual(self, genotype) -> float:
        return NotImplementedError

    @abstractmethod
    def geno2pheno(self, genotype) -> object:
        return NotImplementedError

    def update_robot_xml(self, genotype: np.ndarray):
        return NotImplementedError

    def generate_best_individual_video(self, env, video_name: str, controller: Controller|None = None, n_steps: int = 1000):
        if controller is None:
            controller = self.controller
        rewards_list = []
        observations, info = env.reset()
        controller.reset_controller(1)
        frames = []
        for _ in range(n_steps):
            image = env.render()
            frames.append(image[0] if isinstance(image, tuple) else image)
            action = controller.get_action(observations)
            observations, rewards, terminated, truncated, info = env.step(action)
            rewards_list.append(rewards)
            if terminated:
                break
        print(f"Achieved reward: {np.sum(rewards_list)}")

        import imageio
        imageio.mimsave(video_name, frames, fps=30)  # Set frames per second (fps)

    def visualise_individual(self, genotype, controller: Controller|None = None, n_steps: int = 50000):
        if controller is None:
            controller = self.controller
        self.update_robot_xml(genotype)
        env = self.create_env(render_mode='human')
        rewards_list = []
        observations, info = env.reset()
        controller.reset_controller(1)
        action = controller.get_action(observations)
        for _ in range(n_steps):
            observations, rewards, terminated, truncated, info = env.step(action)
            action = controller.get_action(observations)
            rewards_list.append(rewards)
            if terminated:
                break
        env.close()
        print(f"Achieved reward: {np.sum(rewards_list)}")

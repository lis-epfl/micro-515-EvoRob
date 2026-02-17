import os
import unittest
import gymnasium as gym
import numpy as np

from evorob.utils.filesys import get_project_root
from evorob.algorithms.ga import GA, GA_opts


ROOT_DIR = get_project_root()


class MyTestCase(unittest.TestCase):
    def test_gym(self):
        ENV_NAMES = ["HalfCheetah-v5"]

        for ENV_NAME in ENV_NAMES:
            env = gym.make(
                ENV_NAME,
                render_mode='human')
            rewards = None
            env.reset()
            for step in range(100):
                actions = np.random.uniform(low=-0.3, high=0.3, size=env.action_space.shape[0])
                observations, rewards, terminated, truncated, info = env.step(actions)
                if terminated:
                    break
            env.close()
            self.assertFalse(rewards is None, "Gym environment invalid")

    def f_reversed_ackley(self, x, y):
        return -1 * (
                -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2)))
                - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
                + np.e
                + 20
        )

    def test_functions(self):
        pop_size = 50
        n_params = 2
        results_dir = os.path.join(ROOT_DIR, "results", "TEST")

        ea = GA(pop_size, n_params, GA_opts, results_dir)

        for _ in range(ea.n_gen):
            pop = ea.ask()
            fitnesses_gen = np.empty(ea.n_pop)
            for index, individual in enumerate(pop):
                fit_ind = self.f_reversed_ackley(*individual)
                fitnesses_gen[index] = fit_ind
            ea.tell(pop, fitnesses_gen)
        self.assertLess(-0.1, ea.x_best_so_far.max())


if __name__ == '__main__':
    unittest.main()


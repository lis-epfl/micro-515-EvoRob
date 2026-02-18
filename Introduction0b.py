import xml.etree.ElementTree as xml
from os.path import join
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np

from evorob.algorithms.es import ES, ES_opts
from evorob.utils.filesys import (
    get_distinct_filename,
    get_last_checkpoint_dir,
    get_project_root,
)
from evorob.world.base import World
from evorob.world.robot.controllers.base import Controller
from evorob.world.robot.morphology.passive_walker_robot import PassiveWalkerRobot

""" 
    Morphology optimisation: Passive-Dynamic Walker
"""

ROOT_DIR = get_project_root()
ENV_NAME = "PassiveWalker-v0"


class IdentityController(Controller):

    def get_action(self, state):
        return []


class PassiveWalkerWorld(World):

    def __init__(self):
        self.n_params = 6
        self.temp_dir = TemporaryDirectory()
        self.world_file = join(self.temp_dir.name, "PassiveWalkerEnv.xml")
        self.base_xml_path = join(ROOT_DIR, "evorob", "world", "robot", "assets", "walker_world.xml")
        self.slope_height = np.sin(5 * np.pi / 180) * 5
        self.joint_limits = [[-45, 45], [-150, 0], [-45, 45], [-150, 0], ]
        self.controller = IdentityController()

    def update_robot_xml(self, genotype: np.ndarray):
        points, connectivity_mat = self.geno2pheno(genotype)
        if np.isnan(points).any():
            raise ValueError("Invalid genotype: contains NaN or Inf values.")
        robot = PassiveWalkerRobot(points, connectivity_mat, self.joint_limits, verbose=False)
        robot.xml = robot.define_robot()
        robot.write_xml(self.temp_dir.name)

        # % Defining the Robot environment in MuJoCo #TODO
        world = xml.parse(self.base_xml_path)
        robot_env = world.getroot()

        robot_env.append(xml.Element("include", attrib={"file": "PassiveWalkerRobot.xml"}))
        world_xml = xml.tostring(robot_env, encoding="unicode")
        with open(self.world_file, "w") as f:
            f.write(world_xml)

    def create_env(self, render_mode: str = "rgb_array", **kwargs):
        env = gym.make(
            ENV_NAME,
            robot_path=self.world_file,
            init_z_offset=self.slope_height,
            render_mode=render_mode,
        )
        return env

    def geno2pheno(self, genotype):
        up_l_leg, low_l_leg, l_foot, up_r_leg, low_r_leg, r_foot = genotype

        # Define the 3D coordinates of the relative tree structure
        right_hip_xyz   = np.array([0         ,-0.05    , 0      ])
        right_knee_xyz  = np.array([0         , 0       ,-up_r_leg ]) + right_hip_xyz
        right_ankle_xyz = np.array([0         , 0       ,-low_r_leg]) + right_knee_xyz
        right_toe1_xyz  = np.array([r_foot      ,-0.025   , 0      ]) + right_ankle_xyz
        right_toe2_xyz  = np.array([0         , 0.06    , 0      ]) + right_toe1_xyz

        left_hip_xyz    = np.array([0         , 0.05    , 0       ])
        left_knee_xyz   = np.array([0         , 0       ,-up_l_leg  ]) + left_hip_xyz
        left_ankle_xyz  = np.array([0         , 0       ,-low_l_leg ]) + left_knee_xyz
        left_toe1_xyz   = np.array([l_foot      , 0.025   , 0       ]) + left_ankle_xyz
        left_toe2_xyz   = np.array([0         ,-0.06    , 0       ]) + left_toe1_xyz

        points = np.vstack([right_hip_xyz, right_knee_xyz, right_ankle_xyz, right_toe1_xyz, right_toe2_xyz,
                            left_hip_xyz , left_knee_xyz , left_ankle_xyz , left_toe1_xyz , left_toe2_xyz,])

        # define the type of connections [FIXED ARCHITECTURE]
        connectivity_mat = np.array(
            [[1, np.inf, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, np.inf, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, np.inf, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, np.inf, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, np.inf, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, np.inf, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, np.inf, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, np.inf],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        )
        return points, connectivity_mat

    def evaluate_individual(self, genotype, n_sim_steps=2000):
        try:
            self.update_robot_xml(genotype)
            env = self.create_env()
        except ValueError:
            return -np.inf  # invalid individual
        observations, info = env.reset()
        actions = []
        rewards_list = []
        for _ in range(n_sim_steps):
            observations, rewards, terminated, truncated, info = env.step(actions)
            rewards_list.append(rewards)
            if terminated:
                break
        env.close()
        # TODO: investigate the effects of different fitness functions
        # print(info)
        return sum(rewards_list)


def main():
    #%% Defining environment
    world = PassiveWalkerWorld()
    n_parameters = world.n_params

    #%% Understanding the world
    # TODO: can you improve the genotype - you will also need to modify the PassiveWalkerWorld class!
    genotype = [0.3, 0.2, 0.1, 0.3, 0.2, 0.1]
    world.visualise_individual(genotype)

    results_dir = join(ROOT_DIR, "results", ENV_NAME, "EA")
    results_dir = get_distinct_filename(results_dir)

    # TODO: play with the hyperparameters - these are far from optimal!
    opts = ES_opts.copy()
    opts["min"] = 0
    opts["max"] = 0.5
    opts["num_parents"] = 20
    opts["num_generations"] = 100
    opts["mutation_sigma"] = 0.6
    opts["min_sigma"] = 0.3
    opts["sigma_decay_rate"] = 0.1

    population_size = 100

    ea = ES(population_size, n_parameters, opts, log_every=2, output_dir=results_dir)

    #%% Optimise
    for _ in range(ea.n_gen):
        pop = ea.ask()
        fitnesses_gen = np.empty(ea.n_pop)
        for index, genotype in enumerate(pop):
            fit_ind = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        ea.tell(pop, fitnesses_gen)

    #%% visualise
    checkpoint = get_last_checkpoint_dir(results_dir)
    best_individual = np.load(join(results_dir, checkpoint, "x_best.npy"))
    world.update_robot_xml(best_individual)
    video_name = join(results_dir, "best.mp4")
    print(f"Finished run, generating video [{video_name}]...")
    world.generate_best_individual_video(
        env=world.create_env(),
        video_name=video_name
    )


if __name__ == '__main__':
    main()

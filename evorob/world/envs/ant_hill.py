from os import path
from typing import Dict, Union

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from evorob.utils.geometry import quat2rot

DEFAULT_CAMERA_CONFIG = {
    "distance": 5.,
}


class AntHillEnv(MujocoEnv, utils.EzPickle):
    r"""Gymnasium Ant Benchmark environment.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        robot_path: str,
        frame_skip: int = 5,
        default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1,
        ctrl_cost_weight: float = 0.5,
        cfrc_cost_weight: float = 5e-4,
        main_body: Union[int, str] = 1,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = False,
        pert_force=None,
        **kwargs,
    ):
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            robot_path,
        )

        utils.EzPickle.__init__(
            self,
            xml_file_path,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            cfrc_cost_weight,
            main_body,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            pert_force,
            **kwargs,
        )
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._cfrc_cost_weight = cfrc_cost_weight

        self._main_body = main_body

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        MujocoEnv.__init__(
            self,
            xml_file_path,
            frame_skip,
            observation_space=None,  # needs to be defined after
            default_camera_config=default_camera_config,
            width=832,
            height=496,
            camera_name="track",
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += (
            self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation
        )

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }
        self.body_ids = None
        self.force = None
        self.previous_state = None
        self.stuck = 0
        if pert_force is not None:
            self.body_ids , self.force = pert_force

    def step(self, action):
        xyz_position_before = self.data.body(self._main_body).xpos[:3].copy()
        if self.body_ids is not None:
            self.apply_force()
        self.do_simulation(action, self.frame_skip)
        xyz_position_after = self.data.body(self._main_body).xpos[:3].copy()

        xyz_velocity = (xyz_position_after - xyz_position_before) / self.dt
        x_velocity, y_velocity, z_velocity = xyz_velocity

        forward_reward = x_velocity * self._forward_reward_weight
        healthy_reward = 1
        ctrl_cost = np.sum(action**2)  * self._ctrl_cost_weight
        cfrc_cost = np.sum( self.data.cfrc_ext[1:]**2) * self._cfrc_cost_weight

        #TODO change the reward for hill terrain
        reward = healthy_reward + forward_reward -ctrl_cost -cfrc_cost
        observation = self._get_obs()

        info = {
            "reward_forward": forward_reward,
            "healthy_reward": healthy_reward,
            "ctrl_cost": ctrl_cost,
            "cfrc_cost": cfrc_cost,
            "x_position": self.data.body(self._main_body).xpos[0],
            "y_position": self.data.body(self._main_body).xpos[1],
            "distance_from_origin": np.linalg.norm(self.data.body(self._main_body).xpos[:2]),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "z_velocity": z_velocity,
        }
        terminated = False
        # Check for NaN, Inf, or huge values
        qacc = self.data.qacc
        mask = np.isnan(qacc) | np.isinf(qacc) | (np.abs(qacc) > 1e6)
        # TODO: Re-define termination for slope environment and design appropriate rewards
        if np.any(mask):
            DOF = np.argwhere((np.isnan(qacc)) + (np.isinf(qacc)) + (np.abs(qacc) > 1e6)).squeeze()[0]
            print(ValueError(f'MuJoCo Warning: Nan, Inf or huge value in QACC at DOF {DOF}'))
            terminated = True
        if self.data.qpos[2] < 0.2 or self.data.qpos[2] > 1.0:
            terminated = True
        if terminated:
            info["healthy_reward"] = -10

        self.previous_state = observation

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def torso_upside_down(self,):
        R = self.data.body(self._main_body).xmat.reshape(3, 3)
        torso_z_world = R[:, 2]
        # if dot(torso_z, world_z) < 0 → pointing downward → upside down
        return torso_z_world[2] < 0.0

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate((position, velocity))

    def apply_force(self):
        body_id = self.body_ids
        force = self.force
        pert = self.np_random.uniform(
            low=-0.1, high=0.1, size=3)
        rot = quat2rot([1, *pert])
        force = np.dot(rot, force.reshape(2, 3).T).T.flatten()
        self.data.xfrc_applied[body_id] = force

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale**2
            * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        self.stuck = 0
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }

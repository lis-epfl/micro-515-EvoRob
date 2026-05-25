from os import path
from typing import Dict

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"distance": 5.0}
# Names of throwable sphere joints injected by EvalWorld.update_robot_xml
_SPHERE_JOINT_NAMES = ("joint_sphere_s", "joint_sphere_m", "joint_sphere_l")


class EvalEnv(MujocoEnv, utils.EzPickle):
    """Generic evaluation environment for the MICRO-515 Final Project.

    Loads any robot+terrain XML.  No assumptions are made about the robot
    morphology, controller, or observation/action dimensions — everything is
    inferred at runtime from the MuJoCo model.

    The info dict always exposes the four keys used by the neutral leaderboard
    reward formula (healthy_reward, x_position, ctrl_cost, cfrc_cost).
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        robot_path: str,
        frame_skip: int = 5,
        default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
        ctrl_cost_weight: float = 0.5,
        cfrc_cost_weight: float = 5e-4,
        reset_noise_scale: float = 0.1,
        **kwargs,
    ):
        # Accept an absolute path or resolve relative to this file's directory
        xml_file_path = robot_path if path.isabs(robot_path) else path.join(
            path.dirname(path.realpath(__file__)), robot_path
        )

        utils.EzPickle.__init__(
            self,
            xml_file_path,
            frame_skip,
            default_camera_config,
            ctrl_cost_weight,
            cfrc_cost_weight,
            reset_noise_scale,
            **kwargs,
        )

        self._ctrl_cost_weight = ctrl_cost_weight
        self._cfrc_cost_weight = cfrc_cost_weight
        self._reset_noise_scale = reset_noise_scale

        MujocoEnv.__init__(
            self,
            xml_file_path,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        # Find where robot qpos/qvel ends (sphere joints start after robot joints).
        # Falls back to full nq/nv when no sphere joints are present.
        self._robot_nq, self._robot_nv = self._find_robot_dof_counts()

        obs_size = (self._robot_nq - 2) + self._robot_nv
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )


    def _find_robot_dof_counts(self):
        """Return (robot_nq, robot_nv): qpos/qvel entries that belong to the robot.

        Sphere freejoints are injected after the robot include, so the first
        sphere joint's qposadr/dofadr marks the boundary.
        """
        for jname in _SPHERE_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                return int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid])
        return self.model.nq, self.model.nv

    def step(self, action):
        x_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0]

        x_velocity = (x_after - x_before) / self.dt
        healthy_reward = 1.0
        ctrl_cost = float(np.sum(action ** 2) * self._ctrl_cost_weight)
        cfrc_cost = float(np.sum(self.data.cfrc_ext[1:] ** 2) * self._cfrc_cost_weight)

        reward = healthy_reward + x_velocity - ctrl_cost - cfrc_cost
        observation = self._get_obs()
        terminated = self._is_terminated()

        info = {
            "healthy_reward": -10.0 if terminated else healthy_reward,
            "x_position": float(x_after),
            "ctrl_cost": ctrl_cost,
            "cfrc_cost": cfrc_cost,
            "x_velocity": x_velocity,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _is_terminated(self) -> bool:
        qacc = self.data.qacc
        if bool(np.any(np.isnan(qacc) | np.isinf(qacc) | (np.abs(qacc) > 1e6))):
            return True

        # Lying on back: local z-axis tilted > 120° from upright (cos 120° = -0.5)
        q = self.data.qpos[3:7]  # [w, x, y, z]
        if float(q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2) < -0.5:
            return True

        return False

    def _get_obs(self):
        # Skip robot root xy; exclude sphere joint state (comes after _robot_nq/_robot_nv)
        return np.concatenate((
            self.data.qpos.flat[2:self._robot_nq],
            self.data.qvel.flat[:self._robot_nv],
        ))

    def reset_model(self):
        noise = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(-noise, noise, size=self.model.nq)
        qvel = self.init_qvel + noise ** 2 * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_reset_info(self):
        return {"x_position": float(self.data.qpos[0])}

from pathlib import Path

import numpy as np

from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv


class AntFlatEnvironment(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self, render_mode=None, robot_path: str = "ant_flat_terrain.xml", **kwargs
    ):
        # Load MuJoCo environment in Gymnasium
        # Get path to XML file relative to this module
        xml_file_path = str(Path(__file__).parent / "assets" / robot_path)

        MujocoEnv.__init__(
            self,
            model_path=xml_file_path,
            frame_skip=5,
            observation_space=None,  # needs to be defined after
            render_mode=render_mode,
            default_camera_config={
                "distance": 4.0,
            },
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self._reset_noise_scale: float = 0.1

        # Define observation space.
        # Action space is automatically defined by MuJoCo.
        # Exclude x,y position (first 2 elements of qpos) to match _get_obs()
        obs_size = (self.data.qpos.size - 2) + self.data.qvel.size
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def reset_model(self):
        noise_low = -0.1
        noise_high = 0.1

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def step(self, action):
        torso_body_id = 1
        xy_position_before = self.data.body(torso_body_id).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(torso_body_id).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = self._get_termination()
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_obs(self):
        # TODO: Return observation as concatenation of:
        # - position EXCLUDING x,y: self.data.qpos[2:].flatten() (13 values)
        # - velocity: self.data.qvel.flatten() (14 values)
        # This gives 27 total dimensions, making the task translation-invariant
        # Hint: Use np.concatenate() to combine both arrays
        raise NotImplementedError("TODO: Implement observation function")

    def _get_rew(self, x_velocity: float, action):
        # TODO: Implement reward function with three components:
        # 1. forward_reward = x_velocity * forward_reward_weight (weight=1.0)
        # 2. healthy_reward = healthy_reward_weight (weight=1.0)
        # 3. ctrl_cost = ctrl_cost_weight * sum of squared actions (weight=0.5)
        # Final reward = forward_reward + healthy_reward - ctrl_cost
        # Return: (reward, reward_info_dict)
        raise NotImplementedError("TODO: Implement reward function")

    def _get_termination(self):
        # TODO: Robot should terminate when:
        # - Any value in state is not finite (check with np.isfinite(state).all())
        # - Torso height (state[2]) is below 0.26 or above 1.0
        # Return True if NOT healthy (i.e., should terminate)
        # Hint: Use self.state_vector() to get current state
        raise NotImplementedError("TODO: Implement termination function")

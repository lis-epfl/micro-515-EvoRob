import os
import shutil
import xml.etree.ElementTree as xml
from os.path import join, basename, isfile
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from evorob.utils.filesys import get_last_checkpoint_dir, get_project_root
from evorob.world.base import World
from evorob.world.robot.controllers.base import Controller
from evorob.world.robot.morphology.ant_custom_robot import AntRobot

# Throw schedule: (trigger_x, body_name, y_sign).
# Each entry fires once when robot qpos[0] >= trigger_x during an episode.
# Spheres are thrown from y = y_sign * 7 at height z=4.5, with horizontal
# velocity aimed at bridge centre (y=0).  Order: small → medium → large.
THROW_SCHEDULE = [
    (70,  "sphere_s", +1),
    (74,  "sphere_s", -1),
    (78,  "sphere_s", +1),
    (82,  "sphere_m", -1),
    (86,  "sphere_m", +1),
    (90,  "sphere_m", -1),
    (93,  "sphere_l", +1),
    (96,  "sphere_l", -1),
    (99,  "sphere_l", +1),
]

_SPHERE_PARK = {
    "sphere_s": (75,  0, -10),
    "sphere_m": (90,  0, -10),
    "sphere_l": (105, 0, -10),
}
_SPHERE_JOINTS = {
    "sphere_s": "joint_sphere_s",
    "sphere_m": "joint_sphere_m",
    "sphere_l": "joint_sphere_l",
}

# Second worldbody appended AFTER the robot include so sphere joints follow
# robot joints in qpos/qvel — preserving the original 27-dim observation space.
_SPHERE_WORLDBODY_XML = """<worldbody>
  <body name="sphere_s" pos="75 0 -10">
    <freejoint name="joint_sphere_s"/>
    <inertial pos="0 0 0" mass="1.0" diaginertia="0.025 0.025 0.025"/>
    <geom name="geom_sphere_s" type="sphere" size="0.25"
          rgba="1.0 0.95 0.2 1" friction="0.8 0.5 0.5"
          condim="3" contype="1" conaffinity="1"/>
  </body>
  <body name="sphere_m" pos="90 0 -10">
    <freejoint name="joint_sphere_m"/>
    <inertial pos="0 0 0" mass="5.0" diaginertia="0.32 0.32 0.32"/>
    <geom name="geom_sphere_m" type="sphere" size="0.40"
          rgba="1.0 0.5 0.1 1" friction="0.8 0.5 0.5"
          condim="3" contype="1" conaffinity="1"/>
  </body>
  <body name="sphere_l" pos="105 0 -10">
    <freejoint name="joint_sphere_l"/>
    <inertial pos="0 0 0" mass="15.0" diaginertia="2.16 2.16 2.16"/>
    <geom name="geom_sphere_l" type="sphere" size="0.60"
          rgba="0.85 0.1 0.1 1" friction="0.8 0.5 0.5"
          condim="3" contype="1" conaffinity="1"/>
  </body>
</worldbody>"""

ROOT_DIR = get_project_root()
_EVAL_TERRAIN_XML = join(
    ROOT_DIR, "evorob", "world", "robot", "assets", "eval_terrain.xml"
)
_EVAL_TERRAIN_IMAGE = join(
    ROOT_DIR, "evorob", "world", "robot", "assets", "hilly_hfield.png"
)


class EvalWorld(World):
    """Loads a student's evolved robot and evaluates it on the eval terrain.

    Typical usage
    -------------
    world = EvalWorld()

    # (Optional) swap in your own controller BEFORE calling update_robot_xml:
    # world.set_controller(SO2Controller(input_size=27, output_size=8, hidden_size=8))

    # Option A – you have the robot XML saved from training:
    world.update_robot_xml("/path/to/AntRobot.xml")
    world.geno2pheno(x_best)          # loads controller weights

    # Option B – you only have the checkpoint directory:
    world.load_from_checkpoint("results/final_project")
    """

    def __init__(self):
        self.controller = self._default_controller()
        self.n_weights = self.controller.n_params
        self.n_body_params = 8          # 4 legs × (upper, lower)
        self.n_params = self.n_weights + self.n_body_params

        self.temp_dir = TemporaryDirectory()
        # self.world_file is the tmp copy of eval_terrain.xml with the robot injected
        self.world_file = join(self.temp_dir.name, "eval_terrain.xml")

        self.joint_limits = [
            [-30, 30], [30, 70],
            [-30, 30], [-70, -30],
            [-30, 30], [-70, -30],
            [-30, 30], [30, 70],
        ]
        self.joint_axis = [
            [0, 0, 1], [-1, 1, 0],
            [0, 0, 1], [1, 1, 0],
            [0, 0, 1], [-1, 1, 0],
            [0, 0, 1], [1, 1, 0],
        ]

        # Mirror FinalWorld.sensor_fn — set this if your training used a custom
        # sensor function so the eval run sees the same transformed observations.
        self.sensor_fn = None

    # ------------------------------------------------------------------
    # Controller management
    # ------------------------------------------------------------------

    @staticmethod
    def _default_controller():
        from evorob.world.robot.controllers.mlp_sol import NeuralNetworkController
        return NeuralNetworkController(input_size=27, output_size=8, hidden_size=8)

    def set_controller(self, controller: Controller) -> None:
        """Override the default MLP controller.

        Call this BEFORE update_robot_xml / load_from_checkpoint so that the
        genotype is split at the correct boundary.
        """
        self.controller = controller
        self.n_weights = controller.n_params
        self.n_params = self.n_weights + self.n_body_params
        print(f"Controller set: {type(controller).__name__}  ({controller.n_params} params)")

    # ------------------------------------------------------------------
    # Robot XML injection — same pattern as FinalWorld
    # ------------------------------------------------------------------

    def update_robot_xml(self, final_body_path: str) -> None:
        """Inject a robot XML into the eval terrain template and save to temp dir.

        Copies the robot XML next to the world file so MuJoCo can resolve the
        include relative to self.world_file.

        Args:
            final_body_path: Absolute path to the student's robot body XML
                             (e.g. the AntRobot.xml written by FinalWorld). Must be an absolute path since the world XML will include it with a relative path.
        """
        robot_filename = basename(final_body_path)
        robot_dest_path = join(self.temp_dir.name, robot_filename)
        if os.path.abspath(final_body_path) != os.path.abspath(robot_dest_path):
            shutil.copy2(final_body_path, robot_dest_path)
        shutil.copy2(_EVAL_TERRAIN_IMAGE, join(self.temp_dir.name, basename(_EVAL_TERRAIN_IMAGE)))

        world = xml.parse(_EVAL_TERRAIN_XML)
        robot_env = world.getroot()
        # Robot include comes before sphere worldbody so robot joints appear
        # first in qpos, keeping the original observation dimensionality.
        robot_env.append(xml.Element("include", attrib={"file": robot_filename}))
        robot_env.append(xml.fromstring(_SPHERE_WORLDBODY_XML))
        world_xml = xml.tostring(robot_env, encoding="unicode")
        with open(self.world_file, "w") as f:
            f.write(world_xml)

    # ------------------------------------------------------------------
    # Genotype → controller
    # ------------------------------------------------------------------

    def geno2pheno(self, genotype: np.ndarray) -> None:
        """Pass the controller portion of the genotype to controller.geno2pheno().

        No scaling is applied — the controller's own geno2pheno is responsible
        for any necessary transformation of the raw genotype values.

        The body morphology is NOT regenerated here — call update_robot_xml first
        to provide the robot XML, then call geno2pheno to load the controller.
        """
        self.controller.geno2pheno(genotype[:self.n_weights])

    # ------------------------------------------------------------------
    # One-shot loader from a FinalWorld checkpoint
    # ------------------------------------------------------------------

    def load_from_checkpoint(self, checkpoint_dir: str) -> None:
        """Load robot XML and controller weights from a FinalWorld checkpoint.

        Searches for AntRobot.xml and x_best.npy in the last checkpoint
        generation directory, then falls back to checkpoint_dir itself.
        Works for any body representation — does not assume a fixed genotype structure.

        Args:
            checkpoint_dir: Path to your results directory (e.g. results/final_project).
        """
        last_gen = get_last_checkpoint_dir(checkpoint_dir)
        search_dirs = ([last_gen] if last_gen else []) + [checkpoint_dir]

        def _find(fname):
            for d in search_dirs:
                p = join(d, fname)
                if isfile(p):
                    return p
            return None

        genotype_path = _find("x_best.npy")
        if genotype_path is None:
            raise FileNotFoundError(f"x_best.npy not found in: {checkpoint_dir}")
        genotype = np.load(genotype_path, allow_pickle=True)
        print(f"Loaded genotype: shape={genotype.shape}")

        xml_path = _find("Robot.xml")
        if xml_path is None:
            raise FileNotFoundError(
                f"Robot.xml not found in: {checkpoint_dir}\n"
                "Re-run training with the updated pipeline to save the XML alongside checkpoints."
            )
        print(f"Loaded robot XML: {xml_path}")
        self.update_robot_xml(xml_path)

        self.geno2pheno(genotype)

    # ------------------------------------------------------------------
    # Gymnasium env factory
    # ------------------------------------------------------------------

    def create_env(self, render_mode: str = "rgb_array", **kwargs):
        """Return a ready-to-use EvalEnv-v0 gymnasium environment."""
        import gymnasium as gym
        import evorob.world  # ensures EvalEnv-v0 is registered
        return gym.make(
            "EvalEnv-v0",
            robot_path=self.world_file,
            render_mode=render_mode,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Ball-throwing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _joint_addrs(model, joint_name):
        """Return (qpos_start, dof_start) for a named joint."""
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])

    @staticmethod
    def park_spheres(env):
        """Reset all three throwable spheres to their underground parking spots."""
        model = env.unwrapped.model
        data  = env.unwrapped.data
        for body_name, (px, py, pz) in _SPHERE_PARK.items():
            jname = _SPHERE_JOINTS[body_name]
            try:
                qa, da = EvalWorld._joint_addrs(model, jname)
            except Exception:
                continue  # sphere not present in this model
            data.qpos[qa:qa+3] = [px, py, pz]
            data.qpos[qa+3:qa+7] = [1, 0, 0, 0]  # unit quaternion
            data.qvel[da:da+6] = 0.0
        mujoco.mj_forward(model, data)

    @staticmethod
    def throw_sphere(env, body_name, robot_x, y_sign):
        """Teleport a sphere to throw position and give it lateral velocity.

        Sphere is placed at (robot_x + 2, y_sign*7, 4.5) and thrown with
        velocity (0, y_sign*-9, 0) so it arcs onto the bridge.
        """
        model = env.unwrapped.model
        data  = env.unwrapped.data
        jname = _SPHERE_JOINTS.get(body_name)
        if jname is None:
            return
        try:
            qa, da = EvalWorld._joint_addrs(model, jname)
        except Exception:
            return
        throw_x = robot_x + 2.0
        throw_y = y_sign * 7.0
        throw_z = 4.5
        data.qpos[qa:qa+3] = [throw_x, throw_y, throw_z]
        data.qpos[qa+3:qa+7] = [1, 0, 0, 0]
        data.qvel[da:da+3] = [0.0, y_sign * -9.0, 0.0]
        data.qvel[da+3:da+6] = 0.0
        mujoco.mj_forward(model, data)

    @staticmethod
    def run_throw_step(env, robot_x, pending_throws):
        """Fire any pending throws whose trigger_x has been passed.

        Args:
            env: gymnasium env (must expose .unwrapped.model/.data)
            robot_x: current robot x position
            pending_throws: list of (trigger_x, body_name, y_sign) — mutated in place

        Returns:
            Updated pending_throws list.
        """
        fired = []
        for entry in pending_throws:
            trig_x, body_name, y_sign = entry
            if robot_x >= trig_x:
                EvalWorld.throw_sphere(env, body_name, robot_x, y_sign)
                fired.append(entry)
        for f in fired:
            pending_throws.remove(f)
        return pending_throws

    # ------------------------------------------------------------------
    # Required World abstract methods
    # ------------------------------------------------------------------

    def evaluate_individual(self, genotype: np.ndarray, n_repeats: int = 4,
                            n_steps: int = 500) -> float:
        """Evaluate a genotype on the eval terrain. Returns mean neutral reward."""
        self.geno2pheno(genotype)
        import gymnasium as gym
        import evorob.world

        rewards = []
        for _ in range(n_repeats):
            env = gym.make("EvalEnv-v0", robot_path=self.world_file,
                           max_episode_steps=n_steps)
            self.controller.reset_controller(batch_size=1)
            obs, _ = env.reset()
            total = 0.0
            done = False
            while not done:
                ctrl_obs = self.sensor_fn(obs) if self.sensor_fn is not None else obs
                action = self.controller.get_action(ctrl_obs)
                if action.ndim > 1:
                    action = action.squeeze(0)
                obs, _, terminated, truncated, info = env.step(action)
                total += (
                    float(info.get("healthy_reward", 1.0))
                    + float(info.get("x_position", 0.0))
                    - float(info.get("ctrl_cost", 0.0))
                    - float(info.get("cfrc_cost", 0.0))
                )
                done = terminated or truncated
            rewards.append(total)
            env.close()
        return float(np.mean(rewards))

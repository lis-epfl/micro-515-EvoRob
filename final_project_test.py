"""
MICRO-515 Final Project — Compatibility Check Script
=====================================================
Use this script to verify that your evolved robot is compatible with the
grading pipeline before you submit.

What this script does
---------------------
1.  Loads your best genotype (x_best.npy) and robot XML from a checkpoint
    directory (or from manually specified paths — see Option B below).
2.  Runs the robot on the *dummy* evaluation terrain for N_EPISODES episodes
    (with ball throwing enabled — balls are thrown once the robot reaches
    the bridge, from smaller to larger sphere).
3.  Saves a score file to evaluation_output/.
4.  Records six videos (side / top / pov cameras × two start positions:
    origin and bridge entrance).

Important: the score reported here is computed on a simplified dummy terrain.
It is NOT representative of the actual grading environment, which is hidden
and will only be used during the poster session.  A high score here does not
guarantee a high score on the leaderboard, and vice versa.

Quick-start (recommended)
--------------------------
    python final_project_test.py --best_dir_path results/final_project

The directory must contain x_best.npy (and ideally Robot.xml).
If you used a non-default controller, also set MY_CONTROLLER below.

Option B — supply files manually
----------------------------------
Set ROBOT_XML_PATH to your saved Robot.xml  AND
    GENOTYPE_PATH  to the matching x_best.npy,
then set MY_CONTROLLER if needed.

Submission reminder
--------------------
Always include in your zip:
  - final_project_train.py
  - x_best.npy
  - <your_robot>.xml
  - Your controller source file (even if unchanged from the default)
  - README.md (max 400 words)
"""

import argparse
import copy
import os
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import evorob.world          # registers EvalEnv-v0
import gymnasium as gym

from evorob.world.eval_world import EvalWorld, THROW_SCHEDULE

# ===========================================================================
# STUDENT CONFIGURATION — edit this section
# ===========================================================================

# --- Controller ---
# Set this to the controller you used during training.
# Leave None to use the default (mlp_sol, input=27, output=8, hidden=8).
#
# from evorob.world.robot.controllers.mlp import NeuralNetworkController
# MY_CONTROLLER = NeuralNetworkController(input_size=27, output_size=8, hidden_size=8)
#
# from evorob.world.robot.controllers.so2 import SO2Controller
# MY_CONTROLLER = SO2Controller(input_size=27, output_size=8, hidden_size=8)

MY_CONTROLLER = None

# --- Paths ---
# Option A: directory that contains x_best.npy (recommended)
CHECKPOINT_DIR = "results/final_project"

# Option B: provide the robot XML and genotype as separate files
ROBOT_XML_PATH = None   # e.g. "/abs/path/to/Robot.xml"
GENOTYPE_PATH  = None   # e.g. "/abs/path/to/x_best.npy"

# --- Output ---
OUTPUT_DIR = "evaluation_output"
N_EPISODES = 10     # increase to 256 for the final leaderboard submission
SEED       = 0      # fixed — do NOT change for a fair comparison
MAX_STEPS  = 1000   # fixed — do NOT change

# ===========================================================================


def _neutral_reward(info: dict) -> float:
    """Leaderboard reward: healthy_reward + x_position - ctrl_cost - cfrc_cost."""
    return (
        float(info.get("healthy_reward", 1.0))
        + float(info.get("x_position",   0.0))
        - float(info.get("ctrl_cost",     0.0))
        - float(info.get("cfrc_cost",     0.0))
    )


def _step_episode(env, world, seed_val, bridge_start_x=None):
    """Run one episode, optionally teleporting robot to bridge_start_x after reset.

    Returns (total_reward, frames_dict) where frames_dict maps view key -> list[frame].
    frames_dict keys are "side", "top", "pov" and are populated only when
    env has render_mode="rgb_array".  All views use a tracking free camera
    that follows the robot's x position each step.
    """
    world.controller.reset_controller(batch_size=1)
    obs, _ = env.reset(seed=seed_val)

    # Teleport robot to bridge entrance if requested.
    # Bridge top surface is at z=1.15; shift robot up from ground (z≈0) accordingly.
    if bridge_start_x is not None:
        import mujoco
        env.unwrapped.data.qpos[0] = bridge_start_x
        env.unwrapped.data.qpos[2] += 0.690  # lift from ground level to bridge surface (z=0.690)
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
        obs = env.unwrapped._get_obs()

    # Park all spheres then set up throw schedule
    EvalWorld.park_spheres(env)
    pending = copy.copy(THROW_SCHEDULE)

    total = 0.0
    done  = False
    # Camera presets: (azimuth_deg, elevation_deg, distance, lookat_z_offset)
    # Free-camera mode (camera_id=-1) tracks robot_x each step.
    #   azimuth: 0=behind robot (+x travel dir), 90=from +y side, 270=from -y side
    #   elevation: 90=directly above, 0=horizontal, negative=below horizontal
    _CAM_PRESETS = {
        "side": ( 90, -15, 22, 1.0),   # from +y, slightly above, looking -y
        "top":  (  0, -85, 18, 0.0),   # near-vertical top-down (negative = looking down)
        "pov":  (  0, -10,  9, 1.0),   # from behind robot, looking along +x
    }
    cam_keys = list(_CAM_PRESETS.keys())
    frames = {c: [] for c in cam_keys}
    record = (env.unwrapped.render_mode == "rgb_array")

    while not done:
        robot_x = float(env.unwrapped.data.qpos[0])
        EvalWorld.run_throw_step(env, robot_x, pending)

        if record:
            import mujoco as _mj
            renderer = env.unwrapped.mujoco_renderer
            renderer.camera_id = -1  # use free camera
            viewer = renderer._get_viewer("rgb_array")
            for cam_key, (az, el, dist, lz) in _CAM_PRESETS.items():
                viewer.cam.lookat[:] = [robot_x, 0.0, lz]
                viewer.cam.azimuth   = az
                viewer.cam.elevation = el
                viewer.cam.distance  = dist
                frames[cam_key].append(renderer.render("rgb_array"))

        ctrl_obs = world.sensor_fn(obs) if world.sensor_fn is not None else obs
        action   = world.controller.get_action(ctrl_obs)
        if action.ndim > 1:
            action = action.squeeze(0)
        obs, _, terminated, truncated, info = env.step(action)
        total += _neutral_reward(info)
        done = terminated or truncated

    return total, frames


def run_episodes(world: EvalWorld, n_episodes: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    env = gym.make("EvalEnv-v0", robot_path=world.world_file,
                   max_episode_steps=MAX_STEPS)
    rewards = []

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 2 ** 31))
        total, _ = _step_episode(env, world, ep_seed)
        rewards.append(total)
        print(f"  episode {ep + 1:3d}/{n_episodes}: {total:.2f}")

    env.close()
    return rewards


def record_multi_videos(world: EvalWorld, out_dir: str, seed: int) -> None:
    """Record 6 videos: 3 cameras × 2 start positions (origin, bridge).

    Output files:
        eval_origin_side.mp4   eval_origin_top.mp4   eval_origin_pov.mp4
        eval_bridge_side.mp4   eval_bridge_top.mp4   eval_bridge_pov.mp4
    """
    try:
        import imageio
    except ImportError:
        print("Videos skipped: imageio not installed (pip install imageio[ffmpeg])")
        return

    scenarios = [
        ("origin", None),   # robot starts at x=0
        ("bridge", 70.0),   # robot teleported onto bridge (x=70)
    ]

    for scenario_name, bridge_x in scenarios:
        print(f"\n  Recording scenario '{scenario_name}' …")
        env = gym.make("EvalEnv-v0", robot_path=world.world_file,
                       render_mode="rgb_array", max_episode_steps=MAX_STEPS)
        world.controller.reset_controller(batch_size=1)
        _, frames = _step_episode(env, world, seed, bridge_start_x=bridge_x)
        env.close()

        for cam_key, cam_frames in frames.items():
            if not cam_frames:
                print(f"    {scenario_name}/{cam_key}: no frames")
                continue
            out_path = os.path.join(out_dir, f"eval_{scenario_name}_{cam_key}.mp4")
            try:
                imageio.mimwrite(out_path, cam_frames, fps=20)
                print(f"    Saved: {out_path}  ({len(cam_frames)} frames)")
            except Exception as exc:
                print(f"    {out_path} failed: {exc}")


def save_score(world: EvalWorld, rewards: list, output_dir: str) -> None:
    arr = np.asarray(rewards, dtype=float)
    score_path = os.path.join(output_dir, "evaluation_score.txt")
    with open(score_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MICRO-515 Final Project — Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Controller : {type(world.controller).__name__}"
                f"  ({world.controller.n_params} params)\n")
        f.write(f"Genotype   : {world.n_params} params  "
                f"(controller={world.n_weights}, body={world.n_body_params})\n")
        f.write(f"Reward     : healthy_reward + x_position - ctrl_cost - cfrc_cost\n")
        f.write(f"Ball throw : enabled (spheres thrown after robot reaches bridge)\n\n")
        f.write(f"Mean  : {arr.mean():.2f}\n")
        f.write(f"Std   : {arr.std():.2f}\n")
        f.write(f"Best  : {arr.max():.2f}\n")
        f.write(f"Worst : {arr.min():.2f}\n\n")
        for i, r in enumerate(rewards):
            f.write(f"Episode {i + 1:3d}: {r:10.2f}\n")
    print(f"Score saved: {score_path}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MICRO-515 Final Project — compatibility check against the dummy evaluation terrain."
    )
    parser.add_argument(
        "--best_dir_path",
        default=None,
        metavar="DIR",
        help="Directory containing x_best.npy (and optionally Robot.xml). "
             "Overrides the CHECKPOINT_DIR constant above.",
    )
    args = parser.parse_args()

    checkpoint_dir = args.best_dir_path if args.best_dir_path is not None else CHECKPOINT_DIR

    world = EvalWorld()

    if MY_CONTROLLER is not None:
        world.set_controller(MY_CONTROLLER)

    if ROBOT_XML_PATH is not None and GENOTYPE_PATH is not None:
        # Option B: student provides robot XML and genotype separately
        if not os.path.isfile(ROBOT_XML_PATH):
            raise FileNotFoundError(f"Robot XML not found: {ROBOT_XML_PATH}")
        if not os.path.isfile(GENOTYPE_PATH):
            raise FileNotFoundError(f"Genotype not found: {GENOTYPE_PATH}")
        world.update_robot_xml(ROBOT_XML_PATH)
        genotype = np.load(GENOTYPE_PATH, allow_pickle=True)
        world.controller.geno2pheno(genotype[:world.n_weights])
        print(f"Robot  : {ROBOT_XML_PATH}")
        print(f"Geno   : {GENOTYPE_PATH}  shape={genotype.shape}")
    else:
        # Option A (default): load everything from the checkpoint directory
        world.load_from_checkpoint(checkpoint_dir)

    print(f"\nRunning {N_EPISODES} episodes on the evaluation terrain  (seed={SEED}) …")
    rewards = run_episodes(world, N_EPISODES, SEED)

    arr = np.asarray(rewards, dtype=float)
    print(f"\nResults: mean={arr.mean():.2f} ± {arr.std():.2f}  "
          f"best={arr.max():.2f}  worst={arr.min():.2f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_score(world, rewards, OUTPUT_DIR)
    record_multi_videos(world, OUTPUT_DIR, SEED)

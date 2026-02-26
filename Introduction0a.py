from os.path import join
from time import sleep

import mujoco
import mujoco.viewer

import evorob.algorithms.es

def set_gravity(model, x=0, y=0, z=-9.81):
    print(f"Gravity set to: [{x}, {y}, {z}] m/s²")
    # TODO: Set the gravity vector in the model to the provided x, y, z values


def run_sim(viewer, model, data, seconds: int = 10, dt: float = 0.002):
    step = 0
    n_steps = int(seconds / dt)
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

        step += 1
        sleep(.002)

        if n_steps > 0 and step >= n_steps:
            break


def main():
    xml_path = join('resources', 'exercise0.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    viewer = mujoco.viewer.launch_passive(model, data)

    ds = 0.002

    run_sim(viewer, model, data, seconds=10, dt=ds)

    set_gravity(model, z=-2.2)
    run_sim(viewer, model, data, seconds=10, dt=ds)


if __name__ == '__main__':
    main()

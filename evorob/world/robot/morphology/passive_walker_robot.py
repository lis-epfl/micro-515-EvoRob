import copy
import os.path
import xml.dom.minidom as minidom
import xml.etree.ElementTree as xml

import numpy as np

from evorob.utils.filesys import get_project_root
from evorob.utils.geometry import quat_rel_vecs
from evorob.utils.logging import log

options = {"timestep": "1e-4", "integration": "RK4"}

radius = 0.01
density_rod = (0.05) / (np.pi * radius**2)
density_connect = (0.1) / (4 / 3 * np.pi * radius**3)
properties = {
    "joint": {
        "armature": f"{0.00}",
        "damping": f"{0.01}",
        "axis": "0 -1 0",
        "range": "-150 0",
    },
    "geom": {"margin": f"{0.001}"},
    "rods": {
        "density": f"{density_rod}",
        "size": f"{radius}",
        "rgba": "0.75 0.75 0.75 1.0",
    },
    "connect": {
        "type": "sphere",
        "density": f"{density_connect}",
        "size": f"{radius * 1.1}",
        "rgba": "0.5 0.1 0.1 1.0",
    },
}


class PassiveWalkerRobot:
    def __init__(
        self,
        points,
        connectivity_mat,
        joint_limits=None,
        name: str = "PassiveWalkerRobot",
        props=None,
        fixed_base=False,
        verbose=True,
        z_offset=0.0,
    ):
        if props is None:
            props = properties
        self.properties = props
        self.options = {"timestep": "1e-3", "integrator": "RK4"}
        self.verbose = verbose
        self.xml = None
        self.limbs = []
        self.n_limbs = None
        self.fixed_base = fixed_base
        self.removed_nodes = []
        self.connectivity_mat = connectivity_mat
        self.offset = np.array(
            [
                0,
                0,
                float(props["geom"]["margin"]) * 1.5
                + radius
                - np.min(points[:, 2])
                + z_offset,
            ]
        )
        self.points = points
        self.n_points = points.shape[0]
        self.point_names = [f"p{i}" for i in range(self.n_points)]
        self.name = name
        self.rods = np.argwhere(np.triu(self.connectivity_mat == np.inf))
        self.joints = np.argwhere(np.diag(self.connectivity_mat == 1))
        if joint_limits is None:
            print("WARNING NO JOINT LIMITS SET. DEFAULT TO +- 1")
            joint_limits = [[-1, 1]] * len(self.joints)
        self.joint_limits = joint_limits
        self.motors = np.argwhere(np.diag(self.connectivity_mat > 1))

        self.connect = [
            np.argwhere(self.connectivity_mat[ind] == np.inf).tolist()
            for ind in range(self.n_points)
        ]
        self.identify_structures()

    def write_xml(self, directory: str = "./") -> None:
        xml_string = minidom.parseString(
            xml.tostring(self.xml, encoding="unicode", method="xml")
        ).toprettyxml(indent="    ")
        log(f"Saving xml to: {os.path.join(directory, self.name + '.xml')}", self.verbose)
        with open(os.path.join(directory, self.name + ".xml"), "w") as f:
            f.write(xml_string)
        log("Saved succesfully!", self.verbose)

    def define_robot(self):
        walker_xml = xml.Element("mujoco")
        # xml.SubElement(walker_xml, "options", self.options)

        walker_xml.append(default_setting())
        walker_xml.append(self.define_walker())
        walker_xml.append(self.define_actuators())
        walker_xml.append(self.define_sensor())
        walker_xml.append(self.define_contacts())
        self.xml = walker_xml
        return walker_xml

    def DFSUtil(self, node_list, root_node, visited):
        visited[root_node] = True
        node_list.append(root_node)
        for indices in self.connect[root_node]:
            if indices == []:
                continue
            node = indices[0]
            if not visited[node]:
                node_list = self.DFSUtil(node_list, node, visited)
        return node_list

    def identify_structures(self):
        visited = [False] * self.n_points
        trees = []
        for root_node in range(self.n_points):
            if not visited[root_node]:
                node_list = []
                tree_points = self.DFSUtil(node_list, root_node, visited)
                if len(tree_points) <= 1:
                    self.removed_nodes.append(root_node)
                    continue
                trees.append(tree_points)
        self.n_limbs = len(trees)

        remaining_rods = copy.deepcopy(self.rods).tolist()
        structures = []
        for tree in trees:
            structure_rods = []
            remaining_rods_t = copy.deepcopy(remaining_rods)
            for rod in remaining_rods:
                p1, p2 = rod
                if (p1 in tree) or (p2 in tree):
                    structure_rods.append(rod)
                    remaining_rods_t.remove(rod)
            remaining_rods = remaining_rods_t
            structures.append([structure_rods, tree])
        self.limbs = structures
        return None

    def define_walker(self):
        worldbody_xml = xml.Element("worldbody")
        walker_xml = xml.SubElement(
            worldbody_xml,
            "body",
            attrib={
                "name": "Base",
                "pos": f"{self.offset[0]} {self.offset[1]} {self.offset[2]}",
            },
        )
        if not (self.fixed_base):
            xml.SubElement(walker_xml, "joint", attrib={"type": "free"})

        xml.SubElement(
            walker_xml,
            "geom",
            attrib={
                "type": "sphere",
                "rgba": "0.75 0.75 0.75 1.0",
                "size": "0.05",
                "mass": "0.02",
            },
        )

        for ind, limb in enumerate(self.limbs):
            segments = limb[0]
            segment_cog = np.array(self.points[segments[0][0]])
            parent_xml = walker_xml
            for segment in segments:
                rod_name = f"rod_{segment[0]}.{segment[1]}"
                xyz_from = self.points[segment[0]]
                xyz_to = self.points[segment[1]]
                rel_xyz = (xyz_to - xyz_from) / 2

                segment_cog += rel_xyz
                segment_xml = xml.SubElement(
                    parent_xml,
                    "body",
                    attrib={
                        "name": rod_name,
                        "pos": f"{segment_cog[0]} {segment_cog[1]} {segment_cog[2]}",
                    },
                )

                xml.SubElement(
                    segment_xml,
                    "geom",
                    attrib={
                        "name": rod_name.replace("rod", "geom"),
                        "type": "capsule",
                        "fromto": f"{-rel_xyz[0]} {-rel_xyz[1]} {-rel_xyz[2]} {rel_xyz[0]} {rel_xyz[1]} {rel_xyz[2]}",
                        "size": self.properties["rods"]["size"],
                        "rgba": self.properties["rods"]["rgba"],
                        "density": self.properties["rods"]["density"],
                    },
                )

                quat = quat_rel_vecs([1, 0, 0], xyz_to - xyz_from)
                xml.SubElement(
                    segment_xml,
                    "site",
                    attrib={
                        "name": rod_name.replace("rod", "site"),
                        "pos": f"{0} {0} {0}",
                        "quat": f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}",
                    },
                )

                joint_name = ""
                if segment[0] in self.joints:
                    point_name = self.point_names[segment[0]]
                    joint_ind, _ = np.where(segment[0] == self.joints)
                    joint_name = f"joint_{parent_xml.attrib['name']}={rod_name}"
                    joint_pos = -rel_xyz
                    xml.SubElement(
                        segment_xml,
                        "joint",
                        attrib={
                            "name": joint_name,
                            "pos": f"{joint_pos[0]} {joint_pos[1]} {joint_pos[2]}",
                            "range": f"{self.joint_limits[joint_ind[0]][0]} "
                            f"{self.joint_limits[joint_ind[0]][1]}",
                        },
                    )
                    sphere_xml = xml.SubElement(
                        segment_xml,
                        "body",
                        attrib={
                            "name": point_name,
                            "pos": f"{joint_pos[0]} {joint_pos[1]} {joint_pos[2]}",
                        },
                    )
                    xml.SubElement(
                        sphere_xml,
                        "geom",
                        attrib={
                            "type": self.properties["connect"]["type"],
                            "size": self.properties["connect"]["size"],
                            "rgba": self.properties["connect"]["rgba"],
                            "density": self.properties["connect"]["density"],
                        },
                    )
                    xml.SubElement(sphere_xml, "site", attrib={"name": point_name})
                segment_cog = rel_xyz
                log(f"{rod_name}:\t{np.linalg.norm(xyz_to - xyz_from)} {joint_name}", self.verbose)
                parent_xml = segment_xml

        return worldbody_xml

    def define_actuators(self):
        actuators_xml = xml.Element("actuator")
        for motor in self.motors:
            for limb in self.limbs:
                for base_segment in limb[0]:
                    if base_segment[1] == motor:
                        base_rod_name = f"rod_{base_segment[0]}.{base_segment[1]}"
                        for segment in limb[0]:
                            if segment[0] is not motor:
                                continue
                            rod_name = f"rod_{segment[0]}.{segment[1]}"
                            joint_name = f"joint_{base_rod_name}={rod_name}"
                            xml.SubElement(
                                actuators_xml,
                                "joint",
                                attrib={
                                    "type": self.properties["actuator"]["type"],
                                    "axis": self.properties["actuator"]["axis"],
                                    "joint": joint_name,
                                },
                            )
        return actuators_xml

    def define_sensor(self):
        sensors_xml = xml.Element("sensor")
        for structure in self.limbs:
            rods = structure[0]
            for rod in rods:
                site_name = f"site_{rod[0]}.{rod[1]}"
                log(site_name, self.verbose)
                xml.SubElement(sensors_xml, "accelerometer", attrib={"site": site_name})
                xml.SubElement(sensors_xml, "gyro", attrib={"site": site_name})
                xml.SubElement(sensors_xml, "magnetometer", attrib={"site": site_name})
        return sensors_xml

    def define_contacts(self):
        contact_xml = xml.Element("contact")
        for structure in self.limbs:
            rods = structure[0]
            for rod in rods:
                geom_name = f"geom_{rod[0]}.{rod[1]}"
                xml.SubElement(
                    contact_xml,
                    "pair",
                    attrib={
                        "geom1": geom_name,
                        "geom2": "floor",
                    },
                )
        return contact_xml


def default_setting(props=properties):
    default = xml.Element("default")
    xml.SubElement(
        default,
        "joint",
        attrib={
            "armature": props["joint"]["armature"],
            "damping": props["joint"]["damping"],
            "type": "hinge",
            "range": props["joint"]["range"],
            "axis": props["joint"]["axis"],
        },
    )
    xml.SubElement(
        default,
        "geom",
        attrib={
            "condim": "3",
            "friction": "1.5 1.0 1.0",
            "margin": props["geom"]["margin"],
        },
    )

    motor = xml.SubElement(default, "default", attrib={"class": "motor"})
    xml.SubElement(
        motor,
        "cylinder",
        attrib={
            "gear": "100",
            "ctrllimited": "true",
            "ctrlrange": "-1 1",
        },
    )
    return default


def default_world():
    ROOT_DIR = get_project_root()
    world_xml = xml.parse(
        os.path.join(ROOT_DIR, "src", "world", "robot", "assets", "default_world.xml")
    )
    return world_xml


if __name__ == "__main__":
    ROOT_DIR = get_project_root()

    points = np.array(
        [
            0.0,
            -0.05,
            0.0,
            0,
            -0.05,
            -0.35,
            0,
            -0.05,
            -0.65,
            0.1,
            -0.05,
            -0.65,
            0.1,
            0.02,
            -0.65,
            0.0,
            0.05,
            0.0,
            0,
            0.05,
            -0.35,
            0,
            0.05,
            -0.65,
            0.1,
            0.05,
            -0.65,
            0.1,
            -0.02,
            -0.65,
        ]
    )
    connectivity_mat = np.array(
        [
            [1, np.inf, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, np.inf, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, np.inf, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, np.inf, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, np.inf, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, np.inf, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, np.inf, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, np.inf],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    joint_limits = [
        [-45, 45],
        [-150, 0],
        [-45, 45],
        [-150, 0],
    ]
    test = PassiveWalkerRobot(
        points.reshape(len(points) // 3, 3), connectivity_mat, joint_limits
    )
    tensegrity = xml.Element("mujoco")
    tensegrity.append(default_setting())
    tensegrity.append(test.define_walker())
    tensegrity.append(test.define_sensor())
    tensegrity.append(test.define_actuators())
    tensegrity.append(test.define_contacts())
    test.xml = tensegrity
    test.write_xml()
    world = xml.parse(
        os.path.join(ROOT_DIR, "src", "world", "robot", "assets", "walker_world.xml")
    )
    robot_env = world.getroot()
    # robot_env.attrib
    robot_env.append(xml.Element("include", attrib={"file": "PassiveWalkerRobot.xml"}))
    world_xml = xml.tostring(robot_env, encoding="unicode")
    world_file = "./test.xml"
    with open(world_file, "w") as f:
        f.write(world_xml)

    print("hello")

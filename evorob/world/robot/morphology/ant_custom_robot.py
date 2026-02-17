import copy
import os.path
import xml.dom.minidom as minidom
import xml.etree.ElementTree as xml

import numpy as np

from evorob.utils.filesys import get_project_root
from evorob.utils.geometry import quat_rel_vecs

options = {"timestep": "1e-2", "integration": "RK4"}

radius = 0.08
properties = {
    "joint": {
        "armature": f"{1}",
        "damping": f"{1}",
        "ctrllimited": "true",
        "ctrlrange": "-1.0 1.0",
        "gear": f"{150}",
        "type": "hinge",
        "limited": "true",
    },
    "rods": {
        "size": f"{radius}",
        "conaffinity": "0",
        "condim": "3",
        "density": "5.0",
        "friction": "1 0.5 0.5",
        "margin": "0.01",
        "rgba": "0.8 0.6 0.4 1",
    },
    "geom": {
        "conaffinity": "0",
        "condim": "3",
        "density": "5.0",
        "friction": "1 0.5 0.5",
        "margin": "0.01",
        "rgba": "0.8 0.6 0.4 1",
    },
}


class AntRobot:
    def __init__(
        self,
        points,
        connectivity_mat,
        joint_limits=None,
        joint_axis=None,
        name: str = "AntRobot",
        props=None,
        fixed_base=False,
        verbose=True,
        z_offset=0.0,
    ):
        if props is None:
            props = properties
        self.properties = props
        self.options = options
        self.verbose = verbose
        self.xml = None
        self.limbs = None
        self.n_limbs = None
        self.fixed_base = fixed_base
        self.removed_nodes = []
        self.connectivity_mat = connectivity_mat
        self.offset = np.array([0, 0, 0.75 + z_offset])
        self.points = points
        self.n_points = points.shape[0]
        self.point_names = [f"p{i}" for i in range(self.n_points)]
        self.name = name
        self.rods = np.argwhere(np.triu(self.connectivity_mat == np.inf))
        self.joints = np.argwhere(np.diag(self.connectivity_mat >= 1))
        if joint_limits is None:
            print("WARNING NO JOINT LIMITS SET. DEFAULT TO +- 1")
            joint_limits = [[-30, 30]] * len(self.joints)
        self.joint_limits = joint_limits
        if joint_axis is None:
            print("WARNING NO JOINT LIMITS SET. DEFAULT TO +- 1")
            joint_axis = [[0, 1, 0]] * len(self.joints)
        self.joint_axis = joint_axis
        self.motors = np.argwhere(np.diag(self.connectivity_mat > 1))
        self.motor_refs = []

        self.connect = [
            np.argwhere(self.connectivity_mat[ind] == np.inf).tolist()
            for ind in range(self.n_points)
        ]
        self.identify_structures()

    def write_xml(self, directory: str = "./") -> None:
        xml_string = minidom.parseString(
            xml.tostring(self.xml, encoding="unicode", method="xml")
        ).toprettyxml(indent="    ")
        if self.verbose:
            print("Saving xml to: ", os.path.join(directory, self.name + ".xml"))
        with open(os.path.join(directory, self.name + ".xml"), "w") as f:
            f.write(xml_string)
        if self.verbose:
            print("Saved succesfully!")

    def define_robot(self):
        ant_xml = xml.Element("mujoco")
        # xml.SubElement(ant_xml, "options", self.options)

        ant_xml.append(default_setting())
        ant_xml.append(self.define_ant())
        ant_xml.append(self.define_actuators())
        ant_xml.append(self.define_sensor())
        ant_xml.append(self.define_contacts())
        self.xml = ant_xml
        return ant_xml

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

    def define_ant(self):
        worldbody_xml = xml.Element("worldbody")
        ant_xml = xml.SubElement(
            worldbody_xml,
            "body",
            attrib={
                "name": "Base",
                "pos": f"{self.offset[0]} {self.offset[1]} {self.offset[2]}",
            },
        )
        if not (self.fixed_base):
            xml.SubElement(
                ant_xml,
                "joint",
                attrib={
                    "type": "free",
                    "armature": "0",
                    "damping": "0",
                    "limited": "false",
                    "margin": "0.01",
                    "pos": "0 0 0",
                },
            )
        xml.SubElement(
            ant_xml,
            "geom",
            attrib={
                "type": "sphere",
                "rgba": "0.8 0.6 0.4 1",
                "size": "0.25",
                # "mass": "0.02"
            },
        )
        xml.SubElement(
            ant_xml,
            "camera",
            attrib={
                "name": "track",
                "mode": "trackcom",
                "pos": "0 -3 0.3",
                "xyaxes": "1 0 0 0 0 1",
            },
        )

        for ind, limb in enumerate(self.limbs):
            segments = limb[0]
            segment_cog = np.array(self.points[segments[0][0]])
            parent_xml = ant_xml
            xml.SubElement(
                parent_xml,
                "geom",
                attrib={
                    "name": f"limb{ind}",
                    "type": "capsule",
                    "fromto": f"{0} {0} {0} {segment_cog[0]} {segment_cog[1]} {segment_cog[2]}",
                    "size": self.properties["rods"]["size"],
                },
            )
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
                        "type": "capsule",
                        "fromto": f"{-rel_xyz[0]} {-rel_xyz[1]} {-rel_xyz[2]} {rel_xyz[0]} {rel_xyz[1]} {rel_xyz[2]}",
                        "size": self.properties["rods"]["size"],
                        "name": rod_name.replace("rod", "geom"),
                        # "rgba": self.properties["rods"]["rgba"],
                        # "density": self.properties["rods"]["density"],
                        # "condim": self.properties["rods"]["condim"],
                        # "friction": self.properties["rods"]["friction"],
                        # "margin": self.properties["rods"]["margin"],
                    },
                )

                quat = quat_rel_vecs([1, 0, 0], xyz_to - xyz_from)
                xml.SubElement(
                    segment_xml,
                    "site",
                    attrib={
                        "pos": f"{0} {0} {0}",
                        "quat": f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}",
                        "name": rod_name.replace("rod", "site"),
                    },
                )

                joint_name = ""
                if segment[0] in self.joints:
                    point_name = self.point_names[segment[0]]
                    joint_ind, _ = np.where(segment[0] == self.joints)
                    joint_name = f"joint_{parent_xml.attrib['name']}={rod_name}"
                    for motor in self.motors:
                        if motor == segment[0]:
                            self.motor_refs.append(joint_name)
                            axis = (
                                f"{self.joint_axis[joint_ind[0]][0]} "
                                f"{self.joint_axis[joint_ind[0]][1]} "
                                f"{self.joint_axis[joint_ind[0]][2]}"
                            )
                            # if "Base" in joint_name:
                            #     axis = "0 0 1"

                            break
                    joint_pos = -rel_xyz
                    xml.SubElement(
                        segment_xml,
                        "joint",
                        attrib={
                            "type": self.properties["joint"]["type"],
                            "pos": f"{joint_pos[0]} {joint_pos[1]} {joint_pos[2]}",
                            "axis": axis,
                            "range": f"{self.joint_limits[joint_ind[0]][0]} "
                            f"{self.joint_limits[joint_ind[0]][1]}",
                            "name": joint_name,
                        },
                    )
                    sphere_xml = xml.SubElement(
                        segment_xml,
                        "body",
                        attrib={
                            "pos": f"{joint_pos[0]} {joint_pos[1]} {joint_pos[2]}",
                            "name": point_name,
                        },
                    )
                    xml.SubElement(sphere_xml, "site", attrib={"name": point_name})
                segment_cog = rel_xyz
                if self.verbose:
                    print(
                        f"{rod_name}:\t{np.linalg.norm(xyz_to - xyz_from)} "
                        + joint_name
                    )
                parent_xml = segment_xml

        return worldbody_xml

    def define_actuators(self):
        actuators_xml = xml.Element("actuator")
        for motor_name in self.motor_refs:
            if self.verbose:
                print(motor_name)
            xml.SubElement(
                actuators_xml,
                "motor",
                attrib={
                    "gear": self.properties["joint"]["gear"],
                    "ctrlrange": self.properties["joint"]["ctrlrange"],
                    "ctrllimited": self.properties["joint"]["ctrllimited"],
                    "joint": motor_name,
                },
            )
        return actuators_xml

    def define_sensor(self):
        sensors_xml = xml.Element("sensor")
        for structure in self.limbs:
            rods = structure[0]
            for rod in rods:
                site_name = f"site_{rod[0]}.{rod[1]}"
                if self.verbose:
                    print(site_name)
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
            "limited": props["joint"]["limited"],
            # "type": "hinge",
            # "range": props["joint"]["ctrlrange"]
        },
    )
    xml.SubElement(
        default,
        "geom",
        attrib={
            "condim": props["geom"]["condim"],
            "conaffinity": props["geom"]["conaffinity"],
            "density": props["geom"]["density"],
            "friction": props["geom"]["friction"],
            "margin": props["geom"]["margin"],
            "rgba": props["geom"]["rgba"],
        },
    )

    # motor = xml.SubElement(default, "default", attrib={"class": "motor"})
    # xml.SubElement(motor, "cylinder",
    #                attrib={"gear": "150",
    #                        "ctrllimited": "true",
    #                        "ctrlrange": "-1.0 1.0",
    #                        })
    return default


def default_world():
    ROOT_DIR = get_project_root()
    world_xml = xml.parse(
        os.path.join(ROOT_DIR, "evorob", "world", "robot", "assets", "default_world.xml")
    )
    return world_xml

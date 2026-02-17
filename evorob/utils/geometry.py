import numpy as np


def normalize(v, axis=-1) -> np.ndarray:
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    norm[norm == 0] = 1
    return v / norm


def quat_rel_vecs(v0, v1) -> np.ndarray:
    v0 = normalize(v0)
    v1 = normalize(v1)
    axis = np.cross(v0, v1)
    if np.linalg.norm(axis) == 0:
        if np.dot(v0, v1) > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        else:
            return np.array([0.0, 1.0, 0.0, 0.0])

    axis = normalize(axis)

    # Calculate the angle of rotation
    angle = np.arccos(np.clip(np.dot(v0, v1), -1.0, 1.0))

    # Convert axis-angle to quaternion
    w = np.cos(angle / 2.0)
    x = axis[0] * np.sin(angle / 2.0)
    y = axis[1] * np.sin(angle / 2.0)
    z = axis[2] * np.sin(angle / 2.0)
    return np.array([w, x, y, z])


def quat2rot(quaternion) -> np.ndarray:
    quaternion = normalize(quaternion)
    w, x, y, z = quaternion
    # First row of the rotation matrix
    r00 = 2 * (w * w + x * x) - 1
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)

    # Second row of the rotation matrix
    r10 = 2 * (x * y + w * z)
    r11 = 2 * (w * w + y * y) - 1
    r12 = 2 * (y * z - w * x)

    # Third row of the rotation matrix
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 2 * (w * w + z * z) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
    return rot_matrix


def quat2euler(quaternion) -> tuple:
    quaternion = normalize(quaternion)
    w, x, y, z = quaternion
    ysqr = y * y

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    X = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)

    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    Z = np.arctan2(t3, t4)
    return X, Y, Z


def quat_multiply(q1, q2) -> np.ndarray:
    q1 = normalize(q1)
    q2 = normalize(q2)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def quat_inv(q) -> list:
    w, x, y, z = q
    norm = w*w + x*x + y*y + z*z
    return [w/norm, -x/norm, -y/norm, -z/norm]


def quat2vec2d(q) -> np.ndarray:
    q = normalize(q)
    w, x, y, z = q.T
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([np.cos(yaw), np.sin(yaw)]).T


def transform_points(coordinates, orientation, offset) -> np.ndarray:
    return np.dot(quat2rot(orientation), coordinates.T).T + offset


def are_collinear(P1, P2, P3) -> bool:
    v1 = P2 - P1
    v2 = P3 - P1
    cross_product = np.cross(v1, v2)
    return np.linalg.norm(cross_product) == 0


def rotmat_vecs(v0, v1) -> np.ndarray:
    a = v0 / np.linalg.norm(v0)
    b = v1 / np.linalg.norm(v1)
    v_outer = np.cross(a, b)
    c = np.dot(a, b)
    if c == -1:
        return -np.eye(3)
    s = np.linalg.norm(v_outer)
    kmat = np.array([[0, -v_outer[2], v_outer[1]], [v_outer[2], 0, -v_outer[0]], [-v_outer[1], v_outer[0], 0]])
    if s == 0:
        return np.eye(3)
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def triangular_area(a,b,c) -> float:
    area = 0.25 * np.sqrt((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))
    return area

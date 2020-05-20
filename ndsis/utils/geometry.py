import numpy as np
from typing import Union

def rotation_matrix(
        axis: np.ndarray,
        theta: Union[float, np.ndarray]) -> np.ndarray:
    """Return the rotation matrix associated with counterclockwise rotation
        about the given axis by theta radians.

    Args:
        axis (np.ndarray): :math:`(*, 3)`
            the rotation axis or the array of rotaion axis
        theta (Union[float, np.ndarray]): :math:`(*,)`
            the rotation angle or array of rotation angles

    Returns:
        rotation_matrix (np.ndarray): :math:`(*, 3, 3)`
            the rotation matrix or array of rotation matrices
    """
    axis = np.moveaxis(axis, -1, 0)
    theta = np.asarray(theta)
    axis = axis / np.linalg.norm(axis, axis=0, keepdims=True)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    matrices = np.array(
        [[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    return np.moveaxis(matrices, [0, 1], [-2, -1])


def skew(x: np.ndarray) -> np.ndarray:
    """Calculate a skew symmetric matrix or an array of skew symmetric
        matrices.

    Args:
        x (np.ndarray): (*, 3) a 3D vector or array of 3D vectors.

    Returns:
        skew_symmetric (np.ndarray): (*, 3,3) a matrix or array of skew
            symmetric matrices.
    """
    x1, x2, x3 = np.moveaxis(x, -1, 0)
    zeros = np.zeros_like(x1)
    matrix = np.array([
        [zeros, -x3, x2],
        [x3, zeros, -x1],
        [-x2, x1, zeros]])

    return np.moveaxis(matrix, (0,1), (-2,-1))


def get_rotation_scaling_matrix(origin, destinations):
    """Calculate a rotation and anisotropic scaling matrix to map one vector
        to another by matrix multiplication.

    The matrix is able to rotate the vector and an scale it only in his own
    direction.
    Directions orthogonal to the tensor are not scaled by the matrix.
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/897677#897677
    Origin and destination must be broadcastable.


    Args:
        origin (np.ndarray): (*, 3) a 3D vector or array of 3D vectors to be
            rotated.
        destinations (np.ndarray): (*, 3) a 3D vector or array of 3D vectors to
            be the result of rotation.

    Returns:
        rotation_matrix (np.ndarray): (*, 3,3) a rotation matrix or array of
            rotation matrices.
    """
    height = np.linalg.norm(destinations, axis=-1, keepdims=True)
    unit_destinations = destinations / height

    v = -np.cross(unit_destinations, origin)
    s = np.linalg.norm(v, axis=-1)
    c = np.dot(unit_destinations, origin)
    rotation = np.broadcast_to(
        np.eye(3), (*v.shape[:-1], 3, 3)).copy()

    is_rotation = s != 0
    if isinstance(is_rotation, np.ndarray):
        vx = skew(v[is_rotation])
        cs = c[is_rotation]
        ss = s[is_rotation]
        rotation[is_rotation] += (
            vx
            + np.linalg.matrix_power(vx, 2)
            * ((1 - cs) / (ss * ss))[..., None, None])
    elif is_rotation:
        vx = skew(v)
        rotation += vx + np.dot(vx, vx) * ((1 - c) / (s * s))

    linear = rotation.copy()
    linear[..., 2] *= height

    return linear

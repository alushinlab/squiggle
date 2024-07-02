import numpy as np


def dihedral_angle(pos_1, pos_2, pos_3, pos_4):
    """
    Compute dihedral angle, using the same convention as:
    http://espressomd.org/html/doc/_images/dihedral-angle.pdf
    """
    normalize = lambda v: v / np.linalg.norm(v)

    pos_1 = np.asarray(pos_1)
    pos_2 = np.asarray(pos_2)
    pos_3 = np.asarray(pos_3)
    pos_4 = np.asarray(pos_4)

    vector_12 = pos_2 - pos_1  # 1 -> 2
    vector_23 = pos_3 - pos_2  # 2 -> 3
    vector_34 = pos_4 - pos_3  # 3 -> 4

    # Find the normalized surface normals of the two planes. Their
    # orientations are such that pushing "P1" along normalized_123 and "P4"
    # along normalized_234 would rotate both of them counter-clockwise as
    # seen by the viewer of dihedral-angle.pdf.
    normalized_123 = normalize(np.cross(vector_12, vector_23))
    normalized_234 = normalize(np.cross(vector_23, vector_34))

    normalized_edge = normalize(vector_23)
    vector_axis = np.cross(normalized_234, normalized_123)

    # Apparently "calc_dihedral_angle" in bonded_interactions/dihedral.hpp
    # adds an extra minus sign, effectively flipping the edge direction.
    sin_phi = - np.dot(vector_axis, normalized_edge)
    cos_phi = np.dot(normalized_123, normalized_234)

    phi = np.arctan2(sin_phi, cos_phi)
    return phi


def bond_angle(pos_1, pos_2, pos_3):
    """
    Compute bond angle.
    pos_2 is the position of the vertex
    """
    dist_bottom = np.linalg.norm(pos_2 - pos_1)
    dist_top = np.linalg.norm(pos_3 - pos_2)
    dist_pair = np.linalg.norm(pos_3 - pos_1)
    phi = np.arccos((dist_bottom ** 2 + dist_top ** 2 - dist_pair ** 2) /
                    (2 * dist_bottom * dist_top))
    return phi


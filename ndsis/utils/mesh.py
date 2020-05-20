import plyfile
import numpy as np
from numpy.lib import recfunctions as rfn
from itertools import product
from ndsis.utils.geometry import get_rotation_scaling_matrix


coords_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
color_dtype = [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
vertex_indices_dtype = [('vertex_indices', 'i4', (3,))]
alpha_dtype = [('alpha', 'u1')]
label_dtype = [('label', 'u2')]

ALL_COLORS = np.array([
    [174, 199, 232],  # wall
    [152, 223, 138],  # floor
    [ 31, 119, 180],  # cabinet
    [255, 187, 120],  # bed
    [188, 189,  34],  # chair
    [140,  86,  75],  # sofa
    [255, 152, 150],  # table
    [214,  39,  40],  # door
    [197, 176, 213],  # window
    [148, 103, 189],  # bookshelf
    [196, 156, 148],  # picture
    [ 23, 190, 207],  # counter
    [178,  76,  76],
    [247, 182, 210],  # desk
    [ 66, 188, 102],
    [219, 219, 141],  # curtain
    [140,  57, 197],
    [202, 185,  52],
    [ 51, 176, 203],
    [200,  54, 131],
    [ 92, 193,  61],
    [ 78,  71, 183],
    [172, 114,  82],
    [255, 127,  14],  # refrigerator
    [ 91, 163, 138],
    [153,  98, 156],
    [140, 153, 101],
    [158, 218, 229],  # shower curtain
    [100, 125, 154],
    [178, 127, 135],
    [120, 185, 128],
    [146, 111, 194],
    [ 44, 160,  44],  # toilet
    [112, 128, 144],  # sink
    [ 96, 207, 209],
    [227, 119, 194],  # bathtub
    [213,  92, 176],
    [ 94, 106, 211],
    [ 82,  84, 163],  # otherfurniture
    [100,  85, 144],
    [255, 255, 255]])

ASSOCIATED_COLOR_AND_CLASSNUM = np.array([
    [174, 199, 232, 255,   1],
    [152, 223, 138, 254,   2],
    [ 31, 119, 180, 253,   3],
    [255, 187, 120, 252,   4],
    [188, 189,  34, 251,   5],
    [140,  86,  75, 250,   6],
    [255, 152, 150, 249,   7],
    [214,  39,  40, 248,   8],
    [197, 176, 213, 247,   9],
    [148, 103, 189, 246,  10],
    [196, 156, 148, 245,  11],
    [ 23, 190, 207, 244,  12],
    [247, 182, 210, 242,  14],
    [219, 219, 141, 240,  16],
    [255, 127,  14, 232,  24],
    [158, 218, 229, 228,  28],
    [ 44, 160,  44, 223,  33],
    [112, 128, 144, 222,  34],
    [227, 119, 194, 220,  36],
    [ 82,  84, 163, 217,  39],
    [  0,   0,   0,   0,   0]])


def get_box_indices():
    positions = np.array(list(product([0, 1], repeat=2)))
    x_indices = np.concatenate((
        np.broadcast_to([[0], [1]], (4, 2, 1)),
        np.stack((positions, positions), -2)), axis=-1)
    y_indices = x_indices[:, :, [1, 0, 2]]
    z_indices = x_indices[:, :, [1, 2, 0]]
    all_indices = np.concatenate((x_indices, y_indices, z_indices))
    return all_indices


def get_box_vertex_indices():
    corners = np.arange(8).reshape(2, 2, 2)
    retangles = np.concatenate((
        np.moveaxis(corners, 0, 0).reshape(2, 4),
        np.moveaxis(corners, 1, 0).reshape(2, 4),
        np.moveaxis(corners, 2, 0).reshape(2, 4)))

    triangles = np.concatenate(
        (retangles[:, [1, 2, 3]], retangles[:, [2, 1, 0]]))

    return triangles


VOXEL_VERTEX_INDICES = get_box_vertex_indices()
BOX_INDICES = ..., get_box_indices(), range(3)
# TODO half of faces point in wrong direction


def get_vertex_parts(the_plyfile, parts):
    return np.array(rfn.structured_to_unstructured(
        the_plyfile['vertex'][parts]))


def get_face(the_plyfile):
    return np.stack(the_plyfile['face']['vertex_indices'])


def read_plyfile_from_plypath(plypath):
    the_plyfile = plyfile.PlyData.read(plypath)
    coords = get_vertex_parts(the_plyfile, ['x', 'y', 'z'])
    colors = get_vertex_parts(
        the_plyfile, ['red', 'green', 'blue', 'alpha'])
    face = get_face(the_plyfile)
    return coords, face, colors


def get_colors_from_prediction(prediction):
    prediction = np.array(prediction, dtype=int)
    prediction[prediction < 0] = -1
    return ASSOCIATED_COLOR_AND_CLASSNUM[prediction]


def build_vertex_coords(data, comments=[]):
    structured_array = np.rec.fromarrays(data.T, dtype=coords_dtype)
    return plyfile.PlyElement.describe(
        structured_array, 'vertex', comments=comments)


def build_vertex_colors(data, colors, comments=[]):
    ply_arrays = *data.T, *colors.T
    structured_array = np.rec.fromarrays(
        ply_arrays, dtype=(coords_dtype + color_dtype))
    return plyfile.PlyElement.describe(
        structured_array, 'vertex', comments=comments)


def build_face(indices, comments=[]):
    structured_array = np.rec.fromarrays(
        [indices], dtype=vertex_indices_dtype)
    return plyfile.PlyElement.describe(
        structured_array, 'face', comments=comments)


def combine_vertices_indices(vertices, indices, *others):
    offsets = np.cumsum([0] + [
        len(single_vertices) for single_vertices in vertices[:-1]])
    combined_vertices = np.concatenate(vertices)
    combined_indices = np.concatenate([
        single_indices + single_offset
        for single_indices, single_offset
        in zip(indices, offsets)])

    combined_others = (np.concatenate(other) for other in others)

    return (combined_vertices, combined_indices, *combined_others)


def combine_vertices_indices_retangular(vertices, indices):
    offsets = (np.arange(indices.shape[-3]) * vertices.shape[-2])
    new_vertex_shape = *vertices.shape[:-3], -1, vertices.shape[-1]
    new_index_shape = *indices.shape[:-3], -1, indices.shape[-1]
    combined_vertices = vertices.reshape(new_vertex_shape)
    shifted_indices = indices + offsets[:, None, None]
    combined_indices = shifted_indices.reshape(new_index_shape)
    return combined_vertices, combined_indices


def get_z_cylinder(radius, stacks, slices):
    stacks_range = np.linspace(0, 1, stacks + 1)
    theta_range = np.linspace(0, 2 * np.pi, slices, endpoint=False)
    vertices = np.transpose([
        np.tile(radius * np.cos(theta_range), len(stacks_range)),
        np.tile(radius * np.sin(theta_range), len(stacks_range)),
        np.repeat(stacks_range, len(theta_range))])

    baselines = np.stack(
        (np.arange(slices, 2*slices), np.arange(slices)), axis=1)
    retangles = np.concatenate(
        (baselines, np.roll(baselines, -1, axis=0)), axis=1)
    triangles = np.concatenate(
        (retangles[:, 0:3:1], retangles[:, 3:0:-1]))

    indices = np.concatenate(
        [stack * slices + triangles for stack in range(stacks)])

    return np.array([0, 0, 1]), vertices, indices


def create_cylinder_mesh(
        start, vector, radius=0.3, stacks=8, slices=8):
    transform_origin, raw_vertices, raw_indices = get_z_cylinder(
        radius, stacks, slices)

    linear_transform = get_rotation_scaling_matrix(
        transform_origin, vector)

    transformed = (
        np.tensordot(raw_vertices, linear_transform, (-1, -1)) + start)
    vertices = np.moveaxis(transformed, 0, -2)

    indices = np.broadcast_to(
        raw_indices, (*vertices.shape[:-2], *raw_indices.shape))

    return vertices, indices


def get_box_vertice_endpoints(bbox):
    endpoints = bbox[BOX_INDICES]
    return tuple(np.moveaxis(endpoints, -2, 0))


def get_box_meshes(bbox, radius=0.3, stacks=8, slices=8):
    start, stop = get_box_vertice_endpoints(bbox)
    vector = stop - start
    edgewise_vertices, edgewise_indices = create_cylinder_mesh(
        start, vector, radius, stacks, slices)

    boxwise_vertices, boxwise_indices = combine_vertices_indices_retangular(
        edgewise_vertices, edgewise_indices)

    return boxwise_vertices, boxwise_indices


def get_normal_mesh(
        coords, normals, radius=0.1, stacks=8, slices=8, return_filter=False):
    valid = np.linalg.norm(normals, axis=1) != 0
    valid_normals = normals[valid]
    valid_coords = coords[valid]
    vertices, indices = create_cylinder_mesh(
        valid_coords, valid_normals, radius, stacks, slices)
    valid_vertices, valid_indices = combine_vertices_indices_retangular(
        vertices, indices)
    if return_filter:
        return valid_vertices, valid_indices, valid
    else:
        return valid_vertices, valid_indices


def get_mask_meshes(mask_indices, distance=0.05):
    raw_voxel_vertices = np.array(
        list(product([distance, 1 - distance], repeat=3)))
    voxel_vertices = mask_indices[..., None, :] + raw_voxel_vertices
    voxel_vertex_indices = np.broadcast_to(
        VOXEL_VERTEX_INDICES,
        (*voxel_vertices.shape[:-2], *VOXEL_VERTEX_INDICES.shape))

    return voxel_vertices, voxel_vertex_indices


def get_color(
        labels, vertices, alpha=False, class_num=False, color_factor=None):
    color_selector = [0, 1, 2]
    if alpha:
        color_selector += [3]
    if class_num:
        color_selector += [4]

    new_dims = vertices.shape[1:-1]
    new_shape = *labels.shape, *new_dims, len(color_selector)
    unsqueezer_selector = (labels, *((None,) * len(new_dims)))

    selected_colors = ASSOCIATED_COLOR_AND_CLASSNUM[
        ..., color_selector][unsqueezer_selector]

    broadcasted_colors = np.broadcast_to(
        selected_colors, new_shape)

    if color_factor is not None:
        color_factor_unsqueezer = ..., *(
            [None] * (broadcasted_colors.ndim - color_factor.ndim))
        broadcasted_colors = broadcasted_colors * color_factor[
            color_factor_unsqueezer]

    return broadcasted_colors.reshape(-1, len(color_selector))


def get_dummy_color(color, vertices, color_factor=None):
    color = np.asarray(color)
    new_dims = vertices.shape[:-1]
    new_shape = *new_dims, color.shape[-1]
    unsqueezer = (
        Ellipsis, *((None,) * (vertices.ndim - color.ndim)), slice(None))

    selected_colors = color[unsqueezer]
    broadcasted_colors = np.broadcast_to(selected_colors, new_shape)

    if color_factor is not None:
        color_factor_unsqueezer = ..., *(
            [None] * (broadcasted_colors.ndim - color_factor.ndim))
        broadcasted_colors = broadcasted_colors * color_factor[
            color_factor_unsqueezer]

    return broadcasted_colors.reshape(-1, color.shape[-1])


def get_normal_mesh_full(
        coords, normals, *, color, used_voxel_color=None,
        unused_voxel_color=None, radius=0.1, distance=0.05):
    normal_vertices, normal_indices, valid = get_normal_mesh(
        coords, normals, radius=radius, stacks=4, slices=4, return_filter=True)
    normal_colors = get_dummy_color(color, normal_vertices)
    normal_tuple = normal_vertices, normal_indices, normal_colors

    result_tuples = [normal_tuple]

    if used_voxel_color:
        used_coords = coords[valid]
        used_tuple = get_mask_meshes_full(
            used_coords, color=used_voxel_color, distance=distance)
        result_tuples.append(used_tuple)
    if unused_voxel_color:
        unused_coords = coords[~valid]
        unused_tuple = get_mask_meshes_full(
            used_coords, color=used_voxel_color, distance=distance)
        result_tuples.append(unused_tuple)
    
    return combine_vertices_indices(*zip(*result_tuples))


def get_box_meshes_full(
        bbox, labels=None, color=None, radius=0.3, probability=None,
        stacks=8, slices=8, alpha=False, class_num=False):
    vertices_raw, indices_raw = get_box_meshes(bbox, radius, stacks, slices)
    vertices, indices = combine_vertices_indices_retangular(
            vertices_raw, indices_raw)

    if labels is not None:
        colors = get_color(labels, vertices_raw, alpha, class_num, probability)
    elif color is not None:
        colors = get_dummy_color(color, vertices_raw, probability)
    else:
        colors = None

    if colors is not None:
        return vertices, indices, colors
    else:
        return vertices, indices


def get_mask_meshes_full(
        mask_indices, labels=None, color=None, propabilities=None,
        distance=0.05, alpha=False, class_num=False):
    vertices_raw, indices_raw = get_mask_meshes(mask_indices, distance)
    vertices, indices = combine_vertices_indices_retangular(
            vertices_raw, indices_raw)

    if labels is not None:
        labels = np.array(labels)
        if not len(labels.shape):
            labels = np.broadcast_to(labels, len(mask_indices))
        colors = get_color(
            labels, vertices_raw, alpha, class_num, propabilities)
    elif color is not None:
        colors = get_dummy_color(color, vertices_raw, propabilities)
    else:
        colors = None

    if colors is not None:
        return vertices, indices, colors
    else:
        return vertices, indices


def get_remaining_points(coords, scene_size):
    return (0 < coords).all(-1) & (coords < scene_size).all(-1)


def remove_vertices(remaining_points, coords, face, *args):
    new_indices = np.cumsum(remaining_points, axis=0) - 1
    remaining_face_indicator = remaining_points[face].all(-1)
    remaining_face = new_indices[face[remaining_face_indicator]]
    remaining_coords = coords[remaining_points]
    remaining_args = (other[remaining_points] for other in args)
    return (remaining_coords, remaining_face, *remaining_args)


def project_vertices(coords, coords_projection, coords_shift):
    return coords @ coords_projection + coords_shift


def project_normals(normals, coords_projection):
    return normals @ coords_projection


def remove_project_vertices(
        remaining_points, coords_projection, coords_shift,
        coords, face, colors, normals=None, color_slice=slice(None)):

    if normals is None:
        remaining_coords, remaining_face, remaining_colors = remove_vertices(
            remaining_points, coords, face, colors)
    else:
        (
            remaining_coords, remaining_face, remaining_colors,
            remaining_normals
        ) = remove_vertices(remaining_points, coords, face, colors, normals)

    moved_coords = project_vertices(
        remaining_coords, coords_projection, coords_shift)
    reduced_colors = remaining_colors[:, color_slice]

    if normals is not None:
        projected_normals = project_normals(
            remaining_normals, coords_projection)
        return moved_coords, remaining_face, reduced_colors, projected_normals
    else:
        return moved_coords, remaining_face, reduced_colors


def remove_project_vertices_shape(
        scene_size, coords_projection, coords_shift, coords, face, colors,
        normals=None, color_slice=slice(None)):

    moved_coords = project_vertices(coords, coords_projection, coords_shift)
    reduced_colors = colors[:, color_slice]
    remaining_points = get_remaining_points(moved_coords, scene_size)

    remaining_coords, remaining_face, remaining_colors = remove_vertices(
        remaining_points, moved_coords, face, reduced_colors)

    return remaining_coords, remaining_face, remaining_colors


def build_plyfile(vertices, indices=None, colors=None):
    if colors is None:
        vertex = build_vertex_coords(vertices)
    else:
        vertex = build_vertex_colors(vertices, colors)

    if indices is None:
        full_data = [vertex]
    else:
        face = build_face(indices)
        full_data = [vertex, face]

    return plyfile.PlyData(full_data)

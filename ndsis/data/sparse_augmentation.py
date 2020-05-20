import torch
from math import pi


def ceil_div(a, b):
    return -(-a // b)


def get_coord_distortion_matrix(
        dtype,
        *,
        coord_noise_sigma,
        theta,
        mirror):
    almost_orthonormal = (
        torch.eye(3, dtype=dtype)
        + torch.randn((3, 3), dtype=dtype) * coord_noise_sigma)
    almost_orthonormal[0, 0] *= (
        (torch.randint(0, 2, ()) * 2 - 1)
        if mirror is None
        else (-1 if mirror else 1))

    if theta is None:
        theta = torch.rand((), dtype=dtype) * 2 * pi
    else:
        theta = torch.tensor(theta, dtype=dtype)
        if theta.numel() > 1:
            assert theta.ndim == 1
            index = torch.ones_like(theta).multinomial(1)[0]
            theta = theta[index]

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    rot = torch.tensor([
        [cos_theta,  sin_theta, 0.],
        [-sin_theta, cos_theta, 0.],
        [0.,         0.,        1.]])
    return almost_orthonormal @ rot


def fix_cut_out(discrete_coords, size, shifts):
    start_positions = discrete_coords.new_tensor(-shifts).expand(3)
    moved_coords = discrete_coords - start_positions
    is_inside = ((0 <= discrete_coords) & (discrete_coords < size)).all(-1)
    remaining_coords = moved_coords[is_inside]
    return start_positions, is_inside, remaining_coords


def random_cut_out(discrete_coords, size, max_border):
    num_dims = len(size)
    assert (discrete_coords.shape[1] == num_dims)

    dims_rand_order = torch.multinomial(torch.ones(num_dims), num_dims)
    start_positions = torch.zeros_like(size)
    is_inside = torch.ones_like(discrete_coords[:, 0], dtype=torch.bool)
    inside_coords = discrete_coords.clone()
    for dim in dims_rand_order:
        if not len(inside_coords):
            break
        min_start = inside_coords[:, dim].min() - max_border[dim]
        max_start = (
            inside_coords[:, dim].max() + 1 - size[dim] + max_border[dim])

        if max_start <= min_start:
            start_positions[dim] = min_start
            inside_coords[:, dim] -= start_positions[dim]
        else:
            start_positions[dim] = torch.randint(
                min_start, max_start, (), device=discrete_coords.device)
            inside_coords[:, dim] -= start_positions[dim]
            remaining_inside = (
                (0 <= inside_coords[:, dim]) & 
                (inside_coords[:, dim] < size[dim]))

            inside_coords = inside_coords[remaining_inside]
            is_inside[is_inside] = remaining_inside

    return start_positions, is_inside, inside_coords


def augment_coords(
        coords, *, scale, spatial_size, max_empty_border_size_divisor,
        shift, sub_pixel_offset, **kwargs):
    max_empty_border = (
        (0,) * coords.shape[1]
        if max_empty_border_size_divisor is None else
        tuple(
            single_size // max_empty_border_size_divisor
            for single_size in spatial_size))
    almost_orthonormal = get_coord_distortion_matrix(
        coords.dtype, **kwargs).to(coords.device)
    rot_and_scale = almost_orthonormal * scale
    augmented_coords = coords @ rot_and_scale
    complete_shift = -augmented_coords.min(0).values + (
        torch.rand((3,), dtype=coords.dtype, device=coords.device)
        if sub_pixel_offset is None
        else sub_pixel_offset)
    discrete_coords = (augmented_coords + complete_shift).long()

    if spatial_size is not None:
        spatial_size = discrete_coords.new_tensor(spatial_size).expand(3)

        if shift is None:
            start_positions, is_inside, resulting_coords = random_cut_out(
                discrete_coords, spatial_size, max_empty_border)
        else:
            start_positions, is_inside, resulting_coords = fix_cut_out(
                discrete_coords, spatial_size, shift)
        complete_shift -= start_positions.type(complete_shift.dtype)
    else:
        spatial_size = discrete_coords.max(0).values
        if shift is not None:
            resulting_coords = discrete_coords + shift
            spatial_size += 2 * shift
            complete_shift += shift
        else:
            resulting_coords = discrete_coords
        is_inside = torch.ones_like(discrete_coords[:, 0], dtype=torch.bool)

    augmentation = dict(
        coords_projection=rot_and_scale,
        coords_shift=complete_shift.cpu())

    return (
        resulting_coords, is_inside, spatial_size, augmentation,
        almost_orthonormal)


def augment_single_feature(
        features, is_inside, *, noise_sigma, common_noise, rotation):
    inside_features = features[is_inside]

    if rotation is not None:
        inside_features = inside_features @ rotation

    if noise_sigma:
        feature_noise_shape = (
            inside_features.shape[-1:]
            if common_noise else
            inside_features.shape)
        feature_shift = noise_sigma * torch.randn(
            feature_noise_shape, dtype=features.dtype, device=features.device)

        resulting_features = inside_features + feature_shift
    else:
        feature_shift = inside_features.new_zeros(())
        resulting_features = inside_features

    return resulting_features, feature_shift.cpu()


def augment_features(
        colors, normals, is_inside, resulting_coords, *, color_noise_sigma,
        common_color_noise, normal_noise_sigma, common_normal_noise, rotation,
        use_color, use_ones, use_normal):

    empty_features = colors.new_zeros((len(resulting_coords), 0))

    if use_color:
        resulting_colors, color_shift = augment_single_feature(
            colors, is_inside, noise_sigma=color_noise_sigma,
            common_noise=common_color_noise, rotation=None)
        colors_augmentation = dict(color_shift=color_shift)
    else:
        resulting_colors = empty_features
        colors_augmentation = dict()

    if use_ones:
        ones = colors.new_ones((len(resulting_coords), 1))
    else:
        ones = empty_features

    if use_normal:
        resulting_normals, normals_shift = augment_single_feature(
            normals, is_inside, noise_sigma=normal_noise_sigma,
            common_noise=common_normal_noise, rotation=rotation)
        normals_augmentation = dict(normals_shift=normals_shift)
    else:
        normals_augmentation = dict()
        resulting_normals = empty_features

    features = torch.cat((resulting_colors, ones, resulting_normals), axis=1)
    features_augmentation = {**colors_augmentation, **normals_augmentation}

    return features, features_augmentation


def get_bbox(coords, instance_association_tensor):
    if len(instance_association_tensor):
        instance_coords_list = (
            coords[instance_association]
            for instance_association
            in instance_association_tensor)

        instance_bbox_tensor = torch.stack([
            torch.stack((
                instance_coords.min(0).values,
                instance_coords.max(0).values + 1
                ))
            for instance_coords in instance_coords_list])
    else:
        instance_bbox_tensor = instance_association_tensor.new_zeros(
            (0, 2, coords.shape[1]))

    return instance_bbox_tensor


def get_masks(
        is_inside, instance_ids, semantic_instance_labels,
        instance_cutoff_threshold, label_mapper=None):

    instance_range = torch.arange(
        len(semantic_instance_labels), device=instance_ids.device)

    if label_mapper is not None:
        semantic_instance_labels = label_mapper[semantic_instance_labels]
        keep_ids = semantic_instance_labels >= 0
        semantic_instance_labels = semantic_instance_labels[keep_ids]
        instance_range = instance_range[keep_ids]

    instance_association_tensor = instance_ids == instance_range.unsqueeze(1)

    ratio_inside = torch.tensor([
        is_inside[instance_association].float().mean()
        for instance_association in instance_association_tensor])

    remaining_instances = ratio_inside > instance_cutoff_threshold

    instance_association_tensor = (
        instance_association_tensor[remaining_instances][:, is_inside])
    remaining_semantic_instance_labels = semantic_instance_labels[
        remaining_instances]

    return instance_association_tensor, remaining_semantic_instance_labels


def get_semantic_segmentation_labels(
        is_inside, instance_ids, semantic_instance_labels_raw,
        background_label, label_mapper=None):
    if label_mapper is not None:
        semantic_instance_labels_raw = label_mapper[
            semantic_instance_labels_raw]

    padded_semantic_instance_labels_raw = torch.nn.functional.pad(
        semantic_instance_labels_raw, (0, 1), value=background_label)

    return padded_semantic_instance_labels_raw[instance_ids[is_inside]]


def convert_sample(
        sample, *, spatial_size, instance_cutoff_threshold,
        color_noise_sigma, common_color_noise, normal_noise_sigma,
        common_normal_noise, use_color, use_ones, use_normal,
        additional_bbox_pixel, background_label,
        instance_label_keep=None, instance_label_mapper=None,
        segmentation_label_mapper=None,
        device=None, required_size_factor=None, **kwargs):

    (
        scene_id, coords, colors, normals, instance_ids,
        semantic_instance_labels_raw
    ) = sample

    if device is not None:
        coords = coords.to(device)
        if use_color:
            colors = colors.to(device)
        if use_normal:
            normals = normals.to(device)

    (
        resulting_coords, is_inside, spatial_size, coords_augmentation,
        rotation
    ) = augment_coords(coords, spatial_size=spatial_size, **kwargs)

    spatial_size = spatial_size.cpu()

    features, features_augmentation = augment_features(
        colors, normals, is_inside, resulting_coords, rotation=rotation,
        color_noise_sigma=color_noise_sigma,
        common_color_noise=common_color_noise,
        normal_noise_sigma=normal_noise_sigma,
        common_normal_noise=common_normal_noise,
        use_color=use_color, use_ones=use_ones, use_normal=use_normal)

    augmentation = {
        **coords_augmentation, **features_augmentation,
        'remaining_points': is_inside.cpu()}

    instance_association_tensor, semantic_instance_labels = get_masks(
        is_inside, instance_ids, semantic_instance_labels_raw,
        instance_cutoff_threshold, label_mapper=instance_label_mapper)

    semantic_segmentation_labels = get_semantic_segmentation_labels(
        is_inside, instance_ids, semantic_instance_labels_raw,
        background_label, label_mapper=segmentation_label_mapper)

    instance_bbox_tensor = get_bbox(
        resulting_coords, instance_association_tensor).float()

    if additional_bbox_pixel:
        half_additional_pixels = instance_bbox_tensor.new_tensor(
            [[-additional_bbox_pixel / 2], [additional_bbox_pixel / 2]])
        instance_bbox_tensor += half_additional_pixels

    if required_size_factor is not None:
        spatial_size = required_size_factor * ceil_div(
            spatial_size, required_size_factor)

    return (
        scene_id, resulting_coords, features, instance_bbox_tensor,
        instance_association_tensor, semantic_instance_labels,
        semantic_segmentation_labels, augmentation, spatial_size)

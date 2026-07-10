import random

import numpy as np

from .data_utils import should_apply_condition


def create_flipping_matrix(flip_zyx):
    return np.diag([flip_zyx[0], flip_zyx[1], flip_zyx[2], 1])


def create_translation_matrix(shift_zyx):
    trans_matrix = np.array(
        [
            [1, 0, 0, -shift_zyx[0]],
            [0, 1, 0, -shift_zyx[1]],
            [0, 0, 1, -shift_zyx[2]],
            [0, 0, 0, 1],
        ]
    )
    return trans_matrix


def create_scaling_matrix(scaling_zyx):
    return np.diag(
        [1.0 / scaling_zyx[0], 1.0 / scaling_zyx[1], 1.0 / scaling_zyx[2], 1]
    )


def create_rotation_matrix(theta_zyx):
    theta_z, theta_y, theta_x = theta_zyx

    cz, sz = np.cos(theta_z), np.sin(theta_z)
    cy, sy = np.cos(theta_y), np.sin(theta_y)
    cx, sx = np.cos(theta_x), np.sin(theta_x)

    rotz = np.array(
        [
            [1, 0, 0, 0],  #
            [0, cz, -sz, 0],
            [0, sz, cz, 0],
            [0, 0, 0, 1],
        ]
    )

    roty = np.array(
        [
            [cy, 0, sy, 0],  #
            [0, 1, 0, 0],
            [-sy, 0, cy, 0],
            [0, 0, 0, 1],
        ]
    )

    rotx = np.array(
        [
            [cx, -sx, 0, 0],  #
            [sx, cx, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    return rotz @ roty @ rotx


def random_flipping(random_flip_axis_zyx, is_training, rng=None):
    rng = random if rng is None else rng
    flip_zyx = [1, 1, 1]
    for axis, flag in enumerate(random_flip_axis_zyx):
        if should_apply_condition(0.5, is_training, rng=rng) and flag:
            flip_zyx[axis] = -1
    return create_flipping_matrix(flip_zyx)


def random_scaling(scaling_range_zyx, is_training, rng=None):
    if not is_training:
        return np.eye(4)
    rng = random if rng is None else rng

    def _sample_scale_min_max(min_scale, max_scale):
        return rng.uniform(1 - min_scale, 1 + max_scale)

    random_scale_zyx = [
        _sample_scale_min_max(scaling_range_zyx[axis][0], scaling_range_zyx[axis][1])
        for axis in range(3)
    ]
    return create_scaling_matrix(random_scale_zyx)


def random_rotation(random_rot_deg_zyx, is_training, rng=None):
    if not is_training:
        return np.eye(4)
    rng = random if rng is None else rng

    def _sample_rot(max_rot):
        return np.pi * max_rot / 180.0 * (rng.random() - 0.5) * 2

    theta_zyx = [_sample_rot(random_rot_deg_zyx[axis]) for axis in range(3)]
    return create_rotation_matrix(theta_zyx)


def random_translation(random_shift_rate_zyx, crop_size_zyx, is_training, rng=None):
    if not is_training:
        return np.eye(4)
    rng = random if rng is None else rng

    def _sample_shift(max_shift):
        max_shift = int(round(max_shift))
        return rng.choice(range(-max_shift, max_shift + 1))

    shift_zyx = [
        _sample_shift(crop_size_zyx[axis] * 0.5 * random_shift_rate_zyx[axis])
        for axis in range(3)
    ]
    return create_translation_matrix(shift_zyx)

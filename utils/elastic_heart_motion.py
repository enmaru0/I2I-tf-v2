import os
import numpy as np
import math
from scipy.ndimage.interpolation import zoom

import astra
import elasticdeform
import itertools
import SimpleITK as sitk
import scipy
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation
from PIL import Image
import multiprocessing

from scipy.ndimage import rotate, shift


def apply_rotation(image_2d, angle, rotation_center, max_rotation=[0, 8], heart_rate=60):
    image = np.copy(image_2d)

    """
    Apply oscillating rotation on the 2D image around a point based on heart rate.
    The rotation angle follows a sinusoidal pattern between the max_rotation angles.

    :param image_2d: The 2D image (slice)
    :param angle: The projection angle
    :param rotation_center: The center of rotation (x, y coordinates)
    :param max_rotation: The range of rotation angles
    :param heart_rate: The heart rate of the patient in beats per minute
    :return: The rotated 2D image
    """
    # Convert heart rate from beats per minute to frequency
    frequency = heart_rate / 60.0

    # Calculate rotation angle as a sinusoidal function of the projection angle, modulated by heart rate
    rotation_angle = (max_rotation[1] - max_rotation[0]) * np.sin(angle) + max_rotation[0]

    print("rotation angle ", rotation_angle)

    # Maximum shift distance
    max_shift = max(abs(np.array(rotation_center) - np.array(image.shape) / 2.0))

    # Pad the image before shifting
    padded_image = np.pad(image, pad_width=int(np.ceil(max_shift)), mode='edge')

    # Update rotation center after padding
    rotation_center_padded = np.array(rotation_center) + max_shift

    # Shift the image so that the rotation center is at the origin
    shift_y, shift_x = -rotation_center_padded + np.array(padded_image.shape) / 2.0
    shifted_image = shift(padded_image, [shift_y, shift_x])

    # Apply rotation on the shifted image
    rotated_image = rotate(shifted_image, rotation_angle, reshape=False)

    # Shift the image back to its original position
    unshifted_image = shift(rotated_image, [-shift_y, -shift_x])

    # Crop back to original size
    crop_y, crop_x = (np.array(unshifted_image.shape) - np.array(image.shape)) // 2
    final_image = unshifted_image[crop_y:crop_y + image.shape[0], crop_x:crop_x + image.shape[1]]

    return final_image, rotation_angle


def get_control_point_displacement_matrix_3d(image, mask, magnitude=10, points=100):
    # Parameters
    image_dim_z, image_dim_y, image_dim_x = np.shape(image)
    center = np.argwhere(mask).mean(axis=0)

    displacement_factor = magnitude  # positive for outward, negative for inward

    # Shape of displacement matrix
    num_control_points_z, num_control_points_y, num_control_points_x = points, points, points

    # Calculate the spacing between control points
    spacing_z = image_dim_z / num_control_points_z
    spacing_y = image_dim_y / num_control_points_y
    spacing_x = image_dim_x / num_control_points_x

    # Calculate the displacement matrix
    displacement_matrix = np.zeros((num_control_points_z, num_control_points_y, num_control_points_x, 3))

    # Iterate through the displacement matrix
    for i in range(num_control_points_z):
        for j in range(num_control_points_y):
            for k in range(num_control_points_x):
                # Calculate coordinates of the current control point
                control_point = np.array([i * spacing_z, j * spacing_y, k * spacing_x])

                # Calculate vector from the center to the control point
                vector_to_center = control_point - center

                # Normalize the vector
                normalized_vector = vector_to_center / np.linalg.norm(vector_to_center)

                # Calculate the displacement vector
                displacement_vector = displacement_factor * normalized_vector

                # Assign to the displacement matrix
                displacement_matrix[i, j, k, :] = displacement_vector

    return


def get_control_point_displacement_matrix_2d_radial(mask, magnitude=10, points=100):

    # Parameters
    image_dim_y, image_dim_x = np.shape(mask)
    if (np.sum(mask) !=0):
        center = np.argwhere(mask).mean(axis=0)
    else:
        center = np.asarray([np.shape(mask)[0]/2, np.shape(mask)[1]/2]).astype(np.int16)

    displacement_factor = magnitude  # positive for outward, negative for inward

    # Shape of displacement matrix
    num_control_points_y, num_control_points_x = points, points

    # Calculate the spacing between control points
    spacing_y = image_dim_y / num_control_points_y
    spacing_x = image_dim_x / num_control_points_x

    # Calculate the displacement matrix
    displacement_matrix = np.zeros((2, num_control_points_y, num_control_points_x))

    tolerance = 1e-9
    # Iterate through the displacement matrix
    for j in range(num_control_points_y):
        for k in range(num_control_points_x):
            # Calculate coordinates of the current control point
            control_point = np.array([j * spacing_y, k * spacing_x])

            # Calculate vector from the center to the control point
            vector_to_center = control_point - center

            # Normalize the vector
            normalized_vector = vector_to_center / np.linalg.norm(vector_to_center)

            if (np.linalg.norm(vector_to_center) < tolerance):
                continue

            # Calculate the displacement vector
            displacement_vector = displacement_factor * normalized_vector

            # Assign to the displacement matrix
            displacement_matrix[0, j, k] = displacement_vector[0]
            displacement_matrix[1, j, k] = displacement_vector[1]

    return displacement_matrix

def get_control_point_displacement_matrix_2d_rotation(mask, angle=5, points=100):

    # Parameters
    image_dim_y, image_dim_x = np.shape(mask)
    if (np.sum(mask) !=0):
        center = np.argwhere(mask).mean(axis=0)
    else:
        center = np.asarray([np.shape(mask)[0]/2, np.shape(mask)[1]/2]).astype(np.int16)

    angle_radian = angle * 2 * np.pi / (360)
    cos_theta = np.cos(angle_radian)
    sin_theta = np.sin(angle_radian)

    # Shape of displacement matrix
    num_control_points_y, num_control_points_x = points, points

    # Calculate the spacing between control points
    spacing_y = image_dim_y / num_control_points_y
    spacing_x = image_dim_x / num_control_points_x

    # Calculate the displacement matrix
    displacement_matrix = np.zeros((2, num_control_points_y, num_control_points_x))

    # Iterate through the displacement matrix
    for j in range(num_control_points_y):
        for k in range(num_control_points_x):
            # Calculate coordinates of the current control point
            control_point = np.array([j * spacing_y, k * spacing_x])

            # Calculate the new positions
            k_prime = cos_theta * (control_point[1] - center[1]) - sin_theta * (control_point[0] - center[0]) + center[
                1]
            j_prime = sin_theta * (control_point[1] - center[1]) + cos_theta * (control_point[0] - center[0]) + center[
                0]

            # Calculate the displacement vector
            displacement_vector = (j_prime - control_point[0], k_prime - control_point[1])

            # Assign to the displacement matrix
            displacement_matrix[0, j, k] = displacement_vector[0]
            displacement_matrix[1, j, k] = displacement_vector[1]

    return displacement_matrix

def get_control_point_displacement_matrix_2d_translation(mask, trans_vector=np.asarray([-2, 2]), points=100):

    # Shape of displacement matrix
    num_control_points_y, num_control_points_x = points, points

    # Calculate the displacement matrix
    displacement_matrix = np.zeros((2, num_control_points_y, num_control_points_x))

    # Iterate through the displacement matrix
    for j in range(num_control_points_y):
        for k in range(num_control_points_x):
            # Assign to the displacement matrix
            displacement_matrix[0, j, k] = trans_vector[0]
            displacement_matrix[1, j, k] = trans_vector[1]

    return displacement_matrix


def get_displacement_field(radial_magnitude, rotation_angle, ctrl_points=20, mask=None):
    deformation_mask = binary_dilation(mask, structure=np.ones(shape=[3, 3]))

    displacement = np.zeros((2, ctrl_points, ctrl_points))

    if (radial_magnitude > 0):
        displacement = displacement + get_control_point_displacement_matrix_2d_radial(mask, magnitude=radial_magnitude,
                                                                                      points=ctrl_points)

    if (rotation_angle > 0):
        displacement = displacement + get_control_point_displacement_matrix_2d_rotation(mask, angle=rotation_angle,
                                                                                        points=ctrl_points)

    return displacement


def apply_elastic_deform_2d(image_2d, mask_2d, radial_magnitude, rotation_angle, trans_vector, deform_sigma,
                            ctrl_point=20):


    #deformation_mask = binary_dilation(mask_2d, structure=np.ones(shape=[3, 3]))

    displacement = np.zeros((2, ctrl_point, ctrl_point))

    if (radial_magnitude != 0):
        displacement = get_control_point_displacement_matrix_2d_radial(mask_2d, magnitude=radial_magnitude,
                                                                       points=ctrl_point)

    if (rotation_angle != 0):
        displacement = displacement + get_control_point_displacement_matrix_2d_rotation(mask_2d, angle=rotation_angle,
                                                                                        points=ctrl_point)

    if (np.any(trans_vector) != 0):
        displacement = displacement + get_control_point_displacement_matrix_2d_translation(mask_2d,
                                                                                           trans_vector=trans_vector,
                                                                                           points=ctrl_point)

    [deformed_image, deformed_mask] = elasticdeform.deform_random_grid([image_2d, mask_2d], mask=mask_2d,
                                                                       points=ctrl_point,
                                                                       axis=[(0, 1), (0, 1)], order=[1, 0],
                                                                       sigma=deform_sigma, gauss=deform_sigma,
                                                                       displacement=displacement)

    return deformed_image, displacement


def read_raw_file(filepath):
    raw = np.fromfile(filepath.replace(".hdr", ".raw"), dtype=np.int16)
    hdr = np.loadtxt(filepath, delimiter=" ", dtype=str)
    size = [int(hdr[2]), int(hdr[1]), int(hdr[0])]
    voxel_size = [float(hdr[6]), float(hdr[5]), float(hdr[4])]

    return raw, size, voxel_size


def write_np_array_as_raw_file(array, filename, x=1.0, y=1.0, z=1.0, extension='label', extension2='.raw'):
    if (len(np.shape(array)) == 2):
        array = np.expand_dims(array, axis=0)

    with open(filename.rsplit(".", 1)[0] + '.hdr', 'w') as f:
        f.write(str(array.shape[2]) + ' ' + str(array.shape[1]) + ' ' + str(array.shape[0]) + ' ' + str(2) + ' ' +
                str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + extension)

    array.tofile(filename.rsplit(".", 1)[0] + extension2)

    return


def normalize(img, min=-1024, max=1024):
    image = np.copy(img)
    image = (img - min) / (max - min)

    return image


def normalize_inverse(img, min=-1024, max=1024):
    image = np.copy(img)
    image = (max - min) * img + min

    return image


def oscillating_function(x, max, frequency):
    return max + ((math.sin(frequency * x) + 1) / 2) * (1 - max)


# show an image
def show(img, title=""):
    plt.figure(figsize=(5, 5))
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.clim(0, 1)
    plt.axis('off')
    plt.show()


# create astra projector
def make_projector(angle, vol_geom, nr_detectors):
    proj_geom = astra.create_proj_geom('parallel', 1, nr_detectors, [angle])
    projector_id = astra.creators.create_projector('line', proj_geom, vol_geom)
    return projector_id


# create a random displacement vector grid (fig 5a, 5b and 5c in thesis)
def get_random_displacement(dir_x, dir_y):
    # create 2D Gaussian matrix
    M = 9
    x, y = np.meshgrid(np.linspace(-1, 1, M), np.linspace(-1, 1, M))
    d = np.sqrt(x * x + y * y)

    # random sigma, high value creates uniform motion, low value creates motion focused around single grid point
    sigma = np.random.uniform(0.1, 1.2)
    mu = 0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

    # N x N displacement vector grid
    N = 5
    displacement = np.zeros((2, N, N))

    # take random submatrix to simulate random motion source location
    center_x = int(np.random.uniform(0, N - 1))
    center_y = int(np.random.uniform(0, N - 1))

    # displacement vectors x-values
    displacement[1, :, :] = g[center_x:center_x + N, center_y:center_y + N] * dir_x + 1
    # displacement vectors y-values
    displacement[0, :, :] = g[center_x:center_x + N, center_y:center_y + N] * dir_y + 1

    # clip back the motion vectors to the severity of the original motion direction
    max_movement = np.max((dir_x, dir_y))
    displacement = np.clip(displacement.astype(np.int32), -max_movement, max_movement)

    return displacement


# adapted version of https://github.com/gvtulder/elasticdeform that also returns motion mask
def deform_grid_py(X, displacement, order=3, mode='constant', cval=0.0, crop=None, prefilter=True, axis=None):
    if axis is None:
        axis = tuple(range(X.ndim))
    elif isinstance(axis, int):
        axis = (axis,)

    # compute number of control points in each dimension
    points = [displacement[0].shape[d] for d in range(len(axis))]

    # creates the grid of coordinates of the points of the image (an ndim array per dimension)
    coordinates = np.meshgrid(*[np.arange(X.shape[d]) for d in axis], indexing='ij')
    # creates the grid of coordinates of the points of the image in the "deformation grid" frame of reference
    xi = np.meshgrid(*[np.linspace(0, p - 1, X.shape[d]) for d, p in zip(axis, points)], indexing='ij')

    if crop is not None:
        coordinates = [c[crop] for c in coordinates]
        xi = [x[crop] for x in xi]
        # crop is given only for the axes in axis, convert to all dimensions for the output
        crop = tuple(crop[axis.index(i)] if i in axis else slice(None) for i in range(X.ndim))
    else:
        crop = (slice(None),) * X.ndim
    move_mask = []
    # add the displacement to the coordinates
    for i in range(len(axis)):
        yd = scipy.ndimage.map_coordinates(displacement[i], xi, order=3)
        move_mask.append(yd)
        # adding the displacement
        coordinates[i] = np.add(coordinates[i], yd)

    out = np.zeros(X[crop].shape, dtype=X.dtype)
    # iterate over the non-deformed axes
    iter_axes = [range(X.shape[d]) if d not in axis else [slice(None)]
                 for d in range(X.ndim)]
    for a in itertools.product(*iter_axes):
        scipy.ndimage.map_coordinates(X[a], coordinates, output=out[a],
                                      order=order, cval=cval, mode=mode, prefilter=prefilter)
    return out, np.array(move_mask)


def create_motion_blurr_image(image_3d, mask_3d, z_slice):
    # input: image_3d (numpy array) is a cardiac gated CT image of heart which has data stored as image[z (depth),y (height),x (width)]
    # output: motion_blurr_image (numpy array) is a simulated non-gated image of heart with motion blur

    z_centroid, y_centroid, x_centroid = np.argwhere(mask_3d).mean(axis=0)

    # initialize motion blurr image
    motion_blurr_image = np.zeros(shape=np.shape(image_3d), dtype=np.float64)

    # nr of angles in which the projector will take pictures
    nr_angles = 360

    # the (radian) angles in which the projector will create a projection
    proj_angles = np.linspace(0, np.pi, nr_angles)

    # specify the number of detectors the scanner has, adding 128 to reduce artifacts as image is centered
    nr_detectors = np.max(image_3d.shape[1:2]) + 128

    # get multiple 3D rotates images (1 to 8 degress)

    # Reconstruct one slice at a time
    frequency = 10
    for slice in range(0, np.shape(image_3d)[0], 1):

        print("slice ", slice)

        # initialize data for astra scanner
        vol_geom = astra.creators.create_vol_geom(image_3d.shape[1], image_3d.shape[2])
        sinogram = np.zeros((nr_angles, nr_detectors))

        # start simulation
        for i, proj_angle in enumerate(proj_angles):
            print("proj_angle ", proj_angle)

            # create a new projector for each projection
            projector_id = make_projector(proj_angle, vol_geom, nr_detectors)

            # if(i%frequency == 0):

            # print("proj_angle ", proj_angle)
            # apply rotation motion along z axis
            # based on the projection angle, the 3D image is rotated along z axis
            image_2d, rotation_angle = apply_rotation(image_3d[slice, :, :], proj_angle,
                                                      rotation_center=[y_centroid, x_centroid], max_rotation=[0, 8])

            # if (rotation_angle > 7.0):
            #     plt.imshow(image_2d, cmap='gray')
            #     plt.show()
            #     output = normalize_inverse(image_2d, -1024, 1024)
            #     write_np_array_as_raw_file(output.astype(np.int16), "rotated_image.hdr",x=0.5,y=0.5,z=3.0)
            # else:
            #     image_2d = image_3d[slice,:,:]
            # # apply translation motion
            # # based on the projection angle, the 3D image is translated along x,y,z axis
            # image_3d = apply_translation(image_3d, proj_angle, max_x= [-1,1], max_y=[-1,1], max_z=[-1,1])
            # plt.imshow(image_3d[z_slice, :, :], cmap='gray')
            # plt.show()

            # get slice corresponding to current reconstruction from transformed 3D image
            # image_2d = image_3d[slice,:,:]

            # create artificial sinogram of one angle (add one column for the current projection)
            (sino_id, sino) = astra.creators.create_sino(image_2d, projector_id, returnData=True, gpuIndex=None)

            # store sinograms of all angles
            sinogram[i, :] = sino

            # remove the projector
            astra.projector.delete(projector_id)

        # clean up as the simulation of the scan is done
        astra.projector.clear()

        # create new projector for reconstruction
        proj_geom = astra.create_proj_geom('parallel', 1, nr_detectors, np.linspace(0, np.pi, nr_angles))
        projector_id = astra.creators.create_projector('line', proj_geom, vol_geom)

        # load sinogram data as sinogram object
        sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)

        # create empty reconstruction volume
        reconstruction_id = astra.data2d.create('-vol', vol_geom, data=0)

        # initialize reconstruction algorithm
        alg_cfg = astra.astra_dict('FBP')
        alg_cfg['ProjectorId'] = projector_id
        alg_cfg['ProjectionDataId'] = sinogram_id
        alg_cfg['ReconstructionDataId'] = reconstruction_id
        algorithm_id = astra.algorithm.create(alg_cfg)

        # create reconstruction from sinogram
        astra.algorithm.run(algorithm_id)
        reconstruction = astra.data2d.get(reconstruction_id)

        motion_blurr_image[slice, :, :] = reconstruction

        astra.algorithm.delete(algorithm_id)
        astra.data2d.delete(reconstruction_id)
        astra.data2d.delete(sinogram_id)

    return motion_blurr_image


import imageio


def get_sinogram_2d(image_2d):
    nr_angles = 360

    # the (radian) angles in which the projector will create a projection
    proj_angles = np.linspace(0, np.pi, nr_angles)

    # specify the number of detectors the scanner has, adding 128 to reduce artifacts as image is centered
    nr_detectors = np.max(image_2d.shape[1:2]) + 256

    sinogram = np.zeros((nr_angles, nr_detectors))
    vol_geom = astra.creators.create_vol_geom(image_2d.shape[0], image_2d.shape[1])
    for i, proj_angle in enumerate(proj_angles):
        # create a new projector for each projection
        projector_id = make_projector(proj_angle, vol_geom, nr_detectors)

        # create artificial sinogram of one angle (add one column for the current projection)
        (sino_id, sino) = astra.creators.create_sino(image_2d, projector_id, returnData=True, gpuIndex=None)

        # store sinograms of all angles
        sinogram[i, :] = sino

        # remove the projector
        astra.projector.delete(projector_id)

    return sinogram


def get_moving_sinogram_2d(image, mask, max_radial_magnitude,
                           max_rotation_angle, max_trans_vector, deform_sigma, heart_rate=2.0, rotation_speed=1.0):
    # input: image_3d (numpy array) is a cardiac gated CT image of heart which has data stored as image[z (depth),y (height),x (width)]
    # output: motion_blurr_image (numpy array) is a simulated non-gated image of heart with motion blur

    ratio = heart_rate / rotation_speed

    # nr of angles in which the projector will take pictures
    nr_angles = 360

    # the (radian) angles in which the projector will create a projection
    proj_angles = np.linspace(0, np.pi, nr_angles)

    # specify the number of detectors the scanner has, adding 128 to reduce artifacts as image is centered
    nr_detectors = np.max(image.shape[1:2]) + 256

    # get multiple 3D rotates images (1 to 8 degress)

    # initialize data for astra scanner
    vol_geom = astra.creators.create_vol_geom(image.shape[0], image.shape[1])
    sinogram = np.zeros((nr_angles, nr_detectors))


    # start simulation
    prev_radial_magnitude = None
    prev_rotation_angle = None
    prev_trans_vector = None
    deformed_image = None
    rot_threshold = 1.0#max_rotation_angle/5
    trans_threshold = 1.0  # or any threshold you want for trans_vector
    rad_threshold = 1.0#max_radial_magnitude/5
    images = []
    for i, proj_angle in enumerate(proj_angles):

        # create a new projector for each projection
        projector_id = make_projector(proj_angle, vol_geom, nr_detectors)

        radial_magnitude = max_radial_magnitude * np.sin(proj_angle * ratio)
        rotation_angle = max_rotation_angle * np.sin(proj_angle * ratio)
        trans_vector = max_trans_vector * np.sin(proj_angle * ratio)

        # Check if the change in radial_magnitude, rotation_angle, or any component of trans_vector is substantial
        should_deform = (prev_radial_magnitude is None or abs(radial_magnitude - prev_radial_magnitude) >= rad_threshold) or (
                            prev_rotation_angle is None or abs(rotation_angle - prev_rotation_angle) >= rot_threshold or
                            (prev_trans_vector is None or np.any(np.abs(trans_vector - prev_trans_vector) >= trans_threshold))
        )

        #or(prev_trans_vector is None or np.any(np.abs(trans_vector - prev_trans_vector) >= trans_threshold)
        if should_deform:
            deformed_image, displacement = apply_elastic_deform_2d(image, mask, radial_magnitude=radial_magnitude,
                                                                   rotation_angle=rotation_angle, trans_vector=trans_vector,
                                                                   deform_sigma=deform_sigma, ctrl_point=20)

            #calcium_sum = compute_calcium_sum(deformed_image, mask, threshold=0.5634)
            #print("radial_magnitude ", radial_magnitude, " rotation_angle ", rotation_angle, " trans_vector ", trans_vector, " calcium sum ", calcium_sum)

            # Update the previous values
            prev_radial_magnitude = radial_magnitude
            prev_rotation_angle = rotation_angle
            prev_trans_vector = trans_vector

        images.append(deformed_image)

        # create artificial sinogram of one angle (add one column for the current projection)
        (sino_id, sino) = astra.creators.create_sino(deformed_image, projector_id, returnData=True, gpuIndex=None)

        # store sinograms of all angles
        sinogram[i, :] = sino

        # remove the projector
        astra.projector.delete(projector_id)

    # #Create a GIF
    # with imageio.get_writer('animated.gif', mode='I', fps=40) as writer:
    #     for numpy_image in images:
    #         writer.append_data((numpy_image * 255).astype(np.uint8))

    return sinogram

def create_motion_blurr_image_2d(image_2d, mask_2d, heart_rate, rotation_speed, deform_magnitude):
    # input: image_3d (numpy array) is a cardiac gated CT image of heart which has data stored as image[z (depth),y (height),x (width)]
    # output: motion_blurr_image (numpy array) is a simulated non-gated image of heart with motion blur

    ratio = heart_rate / rotation_speed

    # nr of angles in which the projector will take pictures
    nr_angles = 360

    # the (radian) angles in which the projector will create a projection
    proj_angles = np.linspace(0, np.pi, nr_angles)

    # specify the number of detectors the scanner has, adding 128 to reduce artifacts as image is centered
    nr_detectors = np.max(image_2d.shape[1:2]) + 256

    # get multiple 3D rotates images (1 to 8 degress)

    # initialize data for astra scanner
    vol_geom = astra.creators.create_vol_geom(image_2d.shape[0], image_2d.shape[1])
    sinogram = np.zeros((nr_angles, nr_detectors))

    # start simulation
    prev_sigma = 0
    prev_deformed_image = image_2d
    images = []
    for i, proj_angle in enumerate(proj_angles):

        # create a new projector for each projection
        projector_id = make_projector(proj_angle, vol_geom, nr_detectors)

        sigma = deform_magnitude * np.sin(proj_angle * ratio)

        diff = abs(sigma - prev_sigma)

        if (diff > deform_magnitude / 8):
            print("proj_angle ", proj_angle)
            print("sigma value is ", sigma)
            deformed_image, displacement = apply_elastic_deform_2d(image_2d, mask_2d, sigma=sigma)
            prev_deformed_image = deformed_image
            prev_sigma = sigma
        else:
            deformed_image = prev_deformed_image

        # create artificial sinogram of one angle (add one column for the current projection)
        (sino_id, sino) = astra.creators.create_sino(deformed_image, projector_id, returnData=True, gpuIndex=None)

        # store sinograms of all angles
        sinogram[i, :] = sino

        # remove the projector
        astra.projector.delete(projector_id)

    # # Create a GIF
    # with imageio.get_writer('animated.gif', mode='I', duration=0.01) as writer:
    #     for numpy_image in images:
    #         writer.append_data((numpy_image * 255).astype(np.uint8))

    # clean up as the simulation of the scan is done
    astra.projector.clear()

    # create new projector for reconstruction
    proj_geom = astra.create_proj_geom('parallel', 1, nr_detectors, np.linspace(0, np.pi, nr_angles))
    projector_id = astra.creators.create_projector('line', proj_geom, vol_geom)

    # load sinogram data as sinogram object
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)

    # create empty reconstruction volume
    reconstruction_id = astra.data2d.create('-vol', vol_geom, data=0)

    # initialize reconstruction algorithm
    alg_cfg = astra.astra_dict('FBP')
    alg_cfg['ProjectorId'] = projector_id
    alg_cfg['ProjectionDataId'] = sinogram_id
    alg_cfg['ReconstructionDataId'] = reconstruction_id
    algorithm_id = astra.algorithm.create(alg_cfg)

    # create reconstruction from sinogram
    astra.algorithm.run(algorithm_id)
    reconstruction = astra.data2d.get(reconstruction_id)

    astra.algorithm.delete(algorithm_id)
    astra.data2d.delete(reconstruction_id)
    astra.data2d.delete(sinogram_id)

    return reconstruction

def reconstruct_image_2d(image, sinogram, filter_type="ram-lak"):
    astra.projector.clear()

    # nr of angles in which the projector will take pictures
    nr_angles = 360

    # the (radian) angles in which the projector will create a projection
    proj_angles = np.linspace(0, np.pi, nr_angles)

    # specify the number of detectors the scanner has, adding 128 to reduce artifacts as image is centered
    nr_detectors = np.max(image.shape[1:2]) + 256

    # create new projector for reconstruction
    vol_geom = astra.creators.create_vol_geom(image.shape[0], image.shape[1])
    proj_geom = astra.create_proj_geom('parallel', 1, nr_detectors, np.linspace(0, np.pi, nr_angles))
    projector_id = astra.creators.create_projector('line', proj_geom, vol_geom)

    # load sinogram data as sinogram object
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)

    # create empty reconstruction volume
    reconstruction_id = astra.data2d.create('-vol', vol_geom, data=0)

    # initialize reconstruction algorithm
    alg_cfg = astra.astra_dict('FBP')
    alg_cfg['ProjectorId'] = projector_id
    alg_cfg['ProjectionDataId'] = sinogram_id
    alg_cfg['ReconstructionDataId'] = reconstruction_id
    alg_cfg['FilterType'] = filter_type
    algorithm_id = astra.algorithm.create(alg_cfg)

    # create reconstruction from sinogram
    astra.algorithm.run(algorithm_id)
    reconstruction = astra.data2d.get(reconstruction_id)

    astra.algorithm.delete(algorithm_id)
    astra.data2d.delete(reconstruction_id)
    astra.data2d.delete(sinogram_id)

    return reconstruction


def create_motion_blurr_image_2d_modified(image, mask, spacing,
                                          radial_magnitude,
                                          rotation_angle,
                                          translation_vector,
                                          heart_rate):

    calcium_sum = compute_calcium_sum(image, mask, threshold=0.5634)
    print("image calcium sum ", calcium_sum)

    # Get main deformed image
    rad_magnitude = 0
    rot_angle = 0
    trans_vector = np.asarray([0, 0])
    deform_sigma = 1
    deformed_image, displacement = apply_elastic_deform_2d(image, mask, radial_magnitude=rad_magnitude,
                                                           trans_vector=trans_vector, rotation_angle=rot_angle,
                                                           deform_sigma=deform_sigma)
    sinogram0 = get_sinogram_2d(deformed_image)
    #org_calcium_sum = compute_calcium_sum(deformed_image, mask, threshold=0.5634)
    #print(rad_magnitude, " image calcium sum ", calcium_sum)

    rad_magnitude = radial_magnitude
    rot_angle = rotation_angle
    trans_vector = translation_vector
    deform_sigma = 1
    sinogram_moving= get_moving_sinogram_2d(image, mask, max_radial_magnitude=rad_magnitude,
                                            max_rotation_angle=rot_angle,
                                            max_trans_vector=trans_vector,
                                            deform_sigma=deform_sigma,
                                            heart_rate=heart_rate)

    #sinogram = 0.5 * sinogram0 + 0.25 * sinogram1 + 0.25 * sinogram2
    sinogram = 0.5*sinogram0 + 0.5*sinogram_moving
    reconstruction = reconstruct_image_2d(image, sinogram)

    return reconstruction


def compute_calcium_sum(image, mask, threshold):
    image_ = np.copy(image)
    image_[np.less(image, threshold)] = 0
    image_[np.equal(mask, 0)] = 0

    return np.sum(image_)


def display_displacement(image, displacement_matrix):
    # Interpolate the displacement field to match the size of the image
    factor_y = image.shape[0] / displacement_matrix.shape[2]
    factor_x = image.shape[1] / displacement_matrix.shape[1]
    displacement_slice = zoom(displacement_matrix, (1, factor_y, factor_x))

    # Create a grid for the control points coordinates
    X, Y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))

    # Extract the displacement vectors in the x and y directions
    U = displacement_slice[1, :, :]
    V = displacement_slice[0, :, :]

    # Plot the image slice
    plt.imshow(image, cmap='gray', origin='upper', extent=[0, image.shape[0], 0, image.shape[1]])

    stride = 10

    Y_flipped = image.shape[1] - Y
    V_flipped = -V
    # Overlay the displacement vectors
    plt.quiver(X[::stride, ::stride], Y_flipped[::stride, ::stride], U[::stride, ::stride],
               V_flipped[::stride, ::stride], color='r', angles='xy', scale_units='xy', scale=1)

    # Display the plot
    plt.show()

    return


def get_blurred_image(image, msk, spacing, radial_magnitude, rotation_angle, translation_vector, heart_rate):

    threshold_org = 130
    threshold_rescaled = (130 + 1024) / (2048)  # 0.56640625

    # clip
    image = np.clip(image, a_min=-1024, a_max=1024)

    # normalize between 0 and 1
    image = normalize(image, -1024, 1024)
    print("calcium sum is ", compute_calcium_sum(image, msk, threshold_rescaled))

    motion_blur = np.zeros(shape=np.shape(image))
    for z in range(0, np.shape(image)[0], 1):
        print("slice ", z)
        image_2d = image[z, :, :]
        msk_2d = msk[z, :, :]
        result = create_motion_blurr_image_2d_modified(image_2d, msk_2d, radial_magnitude, rotation_angle, translation_vector, heart_rate)

        motion_blur[z, :, :] = result

    motion_blur = scipy.ndimage.gaussian_filter(motion_blur, sigma=1)

    image = normalize_inverse(image, -1024, 1024)
    motion_blur = normalize_inverse(motion_blur, -1024, 1024)
    diff = image - motion_blur

    return

def process_slice(image_3d, msk_3d, z, spacing,
                  radial_magnitude,
                  rotation_angle,
                  translation_vector,
                  heart_rate):

    print("slice ", z)
    image_2d = \
        image_3d[z, :, :]
    msk_2d = msk_3d[z, :, :]

    result = create_motion_blurr_image_2d_modified(image_2d, msk_2d, spacing, radial_magnitude, rotation_angle, translation_vector, heart_rate)

    return (z, result)

def get_blur_parameters(type):

    if (type == "1"):
        heart_rate = np.random.uniform(2.0, 4.0)
        radial_magnitude = 5.0 * np.random.uniform(2.0, 2.5)
        rotation_angle = 3.0 * np.random.uniform(0.5, 1)
        translation_vector_x = 5.0 * np.random.uniform(-1, 1)
        translation_vector_y = -translation_vector_x
        translation_vector = np.asarray([translation_vector_y, translation_vector_x])
    elif (type == "0"):
        heart_rate = np.random.uniform(2.0, 3.0)
        radial_magnitude = 5.0 * np.random.uniform(1.0, 1.5)
        rotation_angle = 3.0 * np.random.uniform(0.0, 0.5)
        translation_vector_x = 5.0 * np.random.uniform(-0.75, 0.75)
        translation_vector_y = -translation_vector_x
        translation_vector = np.asarray([translation_vector_y, translation_vector_x])
    else:
        heart_rate = np.random.uniform(2.0, 4.0)
        radial_magnitude = 5.0 * np.random.uniform(1.0, 2.5)
        rotation_angle = 3.0 * np.random.uniform(0.0, 1)
        translation_vector_x = 5.0 * np.random.uniform(-1, 1)
        translation_vector_y = -translation_vector_x
        translation_vector = np.asarray([translation_vector_y, translation_vector_x])

    return heart_rate, radial_magnitude, rotation_angle, translation_vector

def get_noise_parameters(type):

    if (type == "1"):
        sigma_xy = np.random.uniform(0.2, 0.4)
        sigma_z = np.random.uniform(0.05, 0.15)
    elif (type == "0"):
        sigma_xy = np.random.uniform(0.4, 1.0)
        sigma_z = np.random.uniform(0.1, 0.3)
    else:
        sigma_xy = np.random.uniform(0.2, 1.0)
        sigma_z = np.random.uniform(0.05, 0.3)

    return sigma_xy, sigma_z

def main():

    #target_spacing = [3.00000, 0.625, 0.625]
    data_dir = "/real/gated/"
    out_data_dir = "/synthetic/non-gated/"

    filepath_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".hdr"):

                if ("gated" in file or "mask" in file or "label" in file):
                    continue

                filepath = os.path.join(root, file)
                filepath_list.append(filepath)

    for filepath in filepath_list:

        print(os.path.basename(filepath))

        # if (os.path.exists(output_filepath)):
        #     print("file already processed")
        #     continue

        msk_filepath = filepath.replace(".hdr", ".msk")
        if not(os.path.exists(msk_filepath)):
            print("mask file does not exists ")
            continue

        if not("ZA-AI001_3_20200821" in filepath):
            continue

        # if os.path.exists(output_filepath):
        #     print("output filepath already exists ")
        #     continue

        threshold_rescaled = (130 + 1024) / (2048)  # 0.56640625

        org_image, size, spacing = read_raw_file(filepath)
        msk = np.fromfile(filepath.replace(".hdr", ".msk"), dtype=np.uint16)
        org_image = np.reshape(org_image, size)
        msk = np.reshape(msk, size)
        msk = 1 * np.right_shift(np.bitwise_and(msk, 1 << 6), 6)

        # clip
        image = np.clip(org_image, a_min=-1024, a_max=1024)

        # normalize between 0 and 1
        image = normalize(image, -1024, 1024)
        #print("calcium sum is ", compute_calcium_sum(image, msk, threshold_rescaled))


        blur_types = ["1", "0"] # 1 stands for high blur, 0 stands for low blur

        for blur_type in blur_types:

            heart_rate, radial_magnitude, rotation_angle, translation_vector = get_blur_parameters(blur_type)
            motion_blur = np.zeros(shape=np.shape(image))

            # experiment
            # heart_rate = 4.0
            # radial_magnitude = 10.0
            # rotation_angle = 0.0
            # translation_vector_x = 5.0
            # translation_vector_y = -5.0

            pool = multiprocessing.Pool()  # Create a pool of worker processes

            results = []
            for z in range(np.shape(image)[0]):
            #for z in range(25,45,1):
                result = pool.apply_async(process_slice, (image, msk, z, spacing,
                                                          radial_magnitude,
                                                          rotation_angle,
                                                          translation_vector,
                                                          heart_rate))
                results.append(result)

            for result in results:
                z, processed_result = result.get()
                motion_blur[z, :, :] = processed_result

            # # # apply random gaussian
            # sigma_xy, sigma_z = get_noise_parameters(noise_type)
            # motion_blur = scipy.ndimage.gaussian_filter(motion_blur, sigma=[sigma_z, sigma_xy, sigma_xy])

            # # # change to target shape
            # temp = zoom(motion_blur, (spacing[0]/target_spacing[0], spacing[1]/target_spacing[1], spacing[1]/target_spacing[1]), order=1)
            # #change back to gated image shape
            # motion_blur = zoom(temp, ((image.shape[0]/temp.shape[0]), (image.shape[1]/temp.shape[1]), (image.shape[1]/temp.shape[1])), order=1)

            motion_blur = normalize_inverse(motion_blur, -1024, 1024)

            if (blur_type == "1"):
                output_filepath = out_data_dir + "/non-gated/high_blur/" + os.path.basename(filepath).replace(".hdr",
                                                                                                    "")
            else:
                output_filepath = out_data_dir + "/non-gated/low_blur/" + os.path.basename(filepath).replace(".hdr",
                                                                                                    "")

            write_np_array_as_raw_file(motion_blur.astype(np.int16), output_filepath, extension="", x=spacing[2], y=spacing[1], z=spacing[0])

            # output_filepath = out_data_dir + "/gated/" + os.path.basename(filepath).replace(".hdr","") + name_extension
            # write_np_array_as_raw_file(org_image.astype(np.int16), output_filepath, extension="", x=spacing[2], y=spacing[1], z=spacing[0])

            # output_filepath = out_data_dir + "/diff/" + os.path.basename(filepath).replace(".hdr","") + name_extension
            # write_np_array_as_raw_file((org_image - motion_blur).astype(np.int16), output_filepath, extension="", x=spacing[2], y=spacing[1], z=spacing[0])
            #
            # output_filepath = out_data_dir + "/diff/" + os.path.basename(filepath).replace(".hdr","") + name_extension
            # with open(output_filepath.rsplit(".", 1)[0] + '.txt', 'w') as f:
            #     f.write(str(heart_rate) + ' ' + str(radial_magnitude) + ' ' + str(rotation_angle) + ' ' + str(2) + ' ' +
            #             str(translation_vector[0]) + ' ' + str(translation_vector[1]))



if __name__ == "__main__":
    main()
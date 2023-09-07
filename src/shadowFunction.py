import numpy as np


def get_weight_pixels(x_ray_loc, x_ray_ang_rad, det_size, det_dist, det_center, pixel_size,
                      diamond1, diamond2, sample_thickness, cone_height, cone_radius1, cone_angle_rad, num):
    """
    Please see the document in the /XRayIonization/Doc/Shadow.pdf for the
    explanation of this function with the definition of the geometry.

    :param x_ray_loc:
    :param x_ray_ang_rad:
    :param det_size:
    :param det_dist:
    :param det_center:
    :param pixel_size:
    :param diamond1:
    :param diamond2:
    :param sample_thickness:
    :param cone_height:
    :param cone_radius1:
    :param cone_angle_rad:
    :param num:
    :return:
    """

    # Create the location of each pixel on the detector
    pixel_position = np.zeros((det_size[0], det_size[1], 3), dtype=np.float64)
    pixel_position[:, :, 0] = (np.arange(0, det_size[0]).astype(np.float64) - det_center[0]) * pixel_size
    pixel_position[:, :, 1] = (np.arange(0, det_size[1]).astype(np.float64) - det_center[1]) * pixel_size
    pixel_position[:, :, 2] = det_dist

    # Create the location along the x-ray trajectory
    x_ray_direction = np.zeros(3, dtype=np.float64)
    x_ray_direction[0] = np.sin(x_ray_ang_rad)  # horizontal is x and is the first axis
    x_ray_direction[1] = 0  # vertical is y and is the second axis
    x_ray_direction[2] = np.cos(x_ray_ang_rad)  # x-ray propagation direction is roughly z and is the last axis

    x_ray_init_loc = np.zeros(3, dtype=np.float64)  # The initial location of the x-ray within the sample
    x_ray_init_loc[0] = x_ray_loc[0]
    x_ray_init_loc[1] = x_ray_loc[1]
    x_ray_init_loc[2] = diamond1

    # Select a few points along the trajectory of the x-ray within the sample
    x_ray_traj = np.zeros((num, 3))  # Sampled points along the trajectory of the x-ray
    x_ray_traj[0, :] = x_ray_init_loc[:]
    x_ray_traj[:, 0] += (np.arange(num).astype(np.float64)) * sample_thickness / num * np.tan(x_ray_ang_rad)
    x_ray_traj[:, 2] += (np.arange(num).astype(np.float64)) * sample_thickness / num
    d_length = sample_thickness / np.tan(x_ray_ang_rad)  # the length between adjacent sampled points.

    # Get the two plane where we have the circles to compare
    plane_height_1 = diamond1 + sample_thickness + diamond2
    plane_height_2 = plane_height_1 + cone_height

    # Get the intersection point of the x-ray trajectory with the two planes
    # x_ray_intersect_1 = (x_ray_init_loc +
    #                     x_ray_direction * ((plane_height_1 - x_ray_init_loc[2]) / np.tan(x_ray_ang_rad)))
    # x_ray_intersect_2 = (x_ray_init_loc +
    #                     x_ray_direction * ((plane_height_2 - x_ray_init_loc[2]) / np.tan(x_ray_ang_rad)))

    weight_holder = np.zeros(det_size, dtype=np.float64)
    # Loop through the sampled points to calculate the weight
    for sample_idx in range(num):
        x_ray_sample = x_ray_traj[sample_idx]

        # Get the intersection between the plane 1 and lines connecting the sample point and the detector pixels.
        ratio1 = (plane_height_1 - x_ray_sample[2]) / (det_dist - x_ray_sample[2])
        x1 = (x_ray_sample[np.newaxis, np.newaxis, :2]
              + ratio1 * (pixel_position[:, :, :2] - x_ray_sample[np.newaxis, np.newaxis, :2]))
        dist1 = np.linalg.norm(x1, axis=-1)

        # Get the intersection between the plane 1 and lines connecting the sample point and the detector pixels.
        ratio2 = (plane_height_2 - x_ray_sample[2]) / (det_dist - x_ray_sample[2])
        x2 = (x_ray_sample[np.newaxis, np.newaxis, :2]
              + ratio2 * (pixel_position[:, :, :2] - x_ray_sample[np.newaxis, np.newaxis, :2]))
        dist2 = np.linalg.norm(x2, axis=-1)

        # Get the pixels that the distance matches both conditions
        weight = np.logical_and(dist1 < cone_radius1, dist2 < cone_radius1 + cone_height * np.tan(cone_angle_rad))

        weight_holder += weight

    # Get the length of the sample contributing to this pixel
    weight_holder *= d_length

    return weight_holder  # the unit is um

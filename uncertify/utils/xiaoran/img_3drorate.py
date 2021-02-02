import scipy.ndimage
import numpy as np
import random

def rotate3d(img):
    dims = img.shape
    assert len(dims)>=3

    angle = np.random(-5,5)
    theta = np.deg2rad(angle)
    tx = 0
    ty = 0

    S, C = np.sin(theta), np.cos(theta)

    # Rotation matrix, angle theta, translation tx, ty
    H = np.array([[C, -S, tx],
                  [S,  C, ty],
                  [0,  0, 1]])

    # Translation matrix to shift the image center to the origin
    r, c = img.shape
    T = np.array([[1, 0, -c / 2.],
                  [0, 1, -r / 2.],
                  [0, 0, 1]])

    # Skew, for perspective
    S = np.array([[1, 0, 0],
                  [0, 1.3, 0],
                  [0, 1e-3, 1]])

    img_rot = transform.homography(img, H)
    #img_rot_center_skew = transform.homography(img, S.dot(np.linalg.inv(T).dot(H).dot(T)))
    return img_rot


def rotate_3d_scipy(img, angles=None):
    dims = img.shape
    assert len(dims) >= 3
    random_angles = 0
    if not angles:
        angle_ax1 = random.uniform(-5, 5)
        angle_ax2 = random.uniform(-5, 5)
        angle_ax3 = random.uniform(-5, 5)
        random_angles=1
    else:
        angle_ax1, angle_ax2, angle_ax3 = angles
    img_rot = scipy.ndimage.interpolation.rotate(img, angle_ax1, mode='nearest',
                                                 axes=(0, 1), reshape=False)
    img_rot = scipy.ndimage.interpolation.rotate(img_rot, angle_ax2, mode='nearest',
                                                 axes=(0, 2), reshape=False)
    # rotate along x-axis
    img_rot = scipy.ndimage.interpolation.rotate(img_rot, angle_ax3, mode='nearest',
                                                 axes=(1, 2), reshape=False)
    if not random_angles:
        return img_rot
    else:
        return img_rot, [angle_ax1, angle_ax2, angle_ax3]




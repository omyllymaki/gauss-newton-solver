import logging
import time

import matplotlib.pyplot as plt
import numpy as np

from gn_solver import GNSolver
from scipy.spatial.transform import Rotation as R

logging.basicConfig(level=logging.INFO)
np.random.seed(42)

NOISE = 0.05


def coeff_to_transform_matrix(coeff):
    ax, ay, az, tx, ty, tz = coeff
    rx = R.from_euler('x', ax).as_matrix()
    ry = R.from_euler('y', ay).as_matrix()
    rz = R.from_euler('z', az).as_matrix()
    rot_mat = rx @ ry @ rz
    trans = np.array([tx, ty, tz])

    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rot_mat
    transform_matrix[:3, 3] = trans

    return transform_matrix


def transform(transform_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1))
    xyz1 = np.hstack((points, ones))
    transformed = (transform_matrix @ xyz1.T).T
    return transformed[:, 0:3]


def func(xyz, coeff):
    t = coeff_to_transform_matrix(coeff)
    return transform(t, xyz)


def residual_func(source, target):
    diff = source - target
    return np.linalg.norm(diff, axis=1)


def main():
    for k in range(16):
        target = np.random.randn(50, 3)
        coeff = np.random.randn(6)
        source = func(target, coeff) + NOISE * np.random.randn(50, 3)

        solver = GNSolver(fit_function=func,
                          residual_function=residual_func,
                          max_iter=100,
                          tolerance_difference=1e-6,
                          tolerance=1e-2)
        init_guess = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        t1 = time.time()
        coeff = solver.fit(source, target, init_guess)
        t2 = time.time()
        duration_ms = 1000*(t2-t1)
        t = coeff_to_transform_matrix(coeff)
        source_transformed = transform(t, source)

        residual = residual_func(source_transformed, target)
        rmse = np.sqrt(np.sum(residual ** 2))

        plt.figure(1)
        plt.subplot(2, 2, 1)
        plt.cla()
        plt.plot(source[:, 0], source[:, 1], "bo", alpha=0.5)
        plt.plot(target[:, 0], target[:, 1], "ro", alpha=0.5)

        plt.subplot(2, 2, 2)
        plt.cla()
        plt.plot(source_transformed[:, 0], source_transformed[:, 1], "bo", alpha=0.5)
        plt.plot(target[:, 0], target[:, 1], "ro", alpha=0.5)

        plt.subplot(2, 1, 2)
        plt.cla()
        plt.plot(solver.rmse_list, "-o")

        plt.figure(2)
        plt.subplot(4, 4, k + 1)
        plt.plot(source_transformed[:, 0], source_transformed[:, 1], "b.", alpha=0.5)
        plt.plot(target[:, 0], target[:, 1], "r.", alpha=0.5)
        plt.title(f"{rmse:0.2f} | {duration_ms:0.1f} ms")

        plt.pause(0.5)

    plt.show()

if __name__ == "__main__":
    main()

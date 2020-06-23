import logging

import matplotlib.pyplot as plt
import numpy as np

from gn_solver import GNSolver

logging.basicConfig(level=logging.INFO)

NOISE = 5
COEFFICIENTS = [-0.001, 0.1, 0.1, 2, 5]


def func(x, coeff):
    return coeff[0] * x ** 3 + coeff[1] * x ** 2 + coeff[2] * x + coeff[3] + coeff[4] * np.sin(x)


def main():
    x = np.arange(1, 100)

    y = func(x, COEFFICIENTS)
    yn = y + NOISE * np.random.randn(len(x))

    solver = GNSolver(fit_function=func, max_iter=100, tolerance_difference=10 ** (-6))
    init_guess = np.random.random(len(COEFFICIENTS))
    _ = solver.fit(x, yn, init_guess)
    fit = solver.get_estimate()
    residual = solver.get_residual()

    plt.plot(x, y, label="Noiseless signal")
    plt.plot(x, yn, label="Noisy signal")
    plt.plot(x, fit, label="Fit")
    plt.plot(x, residual)
    plt.title("Gauss-Newton: curve fitting example")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

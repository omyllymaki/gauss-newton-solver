import logging

import matplotlib.pyplot as plt
import numpy as np

from gn_solver import GNSolver

logging.basicConfig(level=logging.INFO)

NOISE = 3
COEFFICIENTS = [-0.001, 0.1, 0.1, 2, 15]


def func(x, coeff):
    return coeff[0] * x ** 3 + coeff[1] * x ** 2 + coeff[2] * x + coeff[3] + coeff[4] * np.sin(x)


def residual_func(y_fit, y):
    return y_fit - y


def main():
    x = np.arange(1, 100)

    y = func(x, COEFFICIENTS)
    yn = y + NOISE * np.random.randn(len(x))

    solver = GNSolver(fit_function=func,
                      residual_function=residual_func,
                      max_iter=100,
                      tolerance_difference=10 ** (-6))
    init_guess = 1000000 * np.random.random(len(COEFFICIENTS))
    _ = solver.fit(x, yn, init_guess)
    fit = solver.get_estimate()
    residual = solver.get_residual()

    plt.plot(x, y, label="Original, noiseless signal", linewidth=2)
    plt.plot(x, yn, label="Noisy signal", linewidth=2)
    plt.plot(x, fit, label="Fit", linewidth=2)
    plt.plot(x, residual, label="Residual", linewidth=2)
    plt.title("Gauss-Newton: curve fitting example")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

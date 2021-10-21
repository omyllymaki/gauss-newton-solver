import logging
from typing import Callable

import numpy as np
from numpy.linalg import pinv

logger = logging.getLogger(__name__)


class GNSolver:
    """
    Gauss-Newton solver.

    Given response vector y, dependent variable x and fit function f, 
    Minimize sum(residual^2) where residual = f(x, coefficients) - y.
    """

    def __init__(self,
                 fit_function: Callable,
                 residual_function: Callable,
                 update_step_size: float = 1.0,
                 min_iter: int = 1,
                 max_iter: int = 1000,
                 jacobian_step_size=1e-9,
                 tolerance_difference: float = 1e-16,
                 tolerance: float = 1e-9,
                 init_guess: np.ndarray = None,
                 ):
        """
        :param fit_function: Function that needs to be fitted; y_estimate = fit_function(x, coefficients).
        :param residual_function: Function that calculates residuals; residuals = residual_function(y_estimate, y).
        :param update_step_size: Determines steps size (fraction) for coefficient update; should be in range [0, 1].
        :param min_iter: Minimum number of iterations for optimization.
        :param max_iter: Maximum number of iterations for optimization.
        :param jacobian_step_size: Step size for numerical Jacobian calculation.
        :param tolerance_difference: Terminate iteration if RMSE difference between iterations smaller than tolerance.
        :param tolerance: Terminate iteration if RMSE is smaller than tolerance.
        :param init_guess: Initial guess for coefficients.
        """
        self.fit_function = fit_function
        self.residual_function = residual_function
        self.update_step_size = update_step_size
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.epsilon = jacobian_step_size
        self.tolerance_difference = tolerance_difference
        self.tolerance = tolerance
        self.coefficients = None
        self.x = None
        self.y = None
        self.init_guess = None
        if init_guess is not None:
            self.init_guess = init_guess

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            init_guess: np.ndarray = None) -> np.ndarray:
        """
        Fit coefficients by minimizing RMSE.

        :param x: Independent variable.
        :param y: Response vector.
        :param init_guess: Initial guess for coefficients.
        :return: Fitted coefficients.
        """

        self.x = x
        self.y = y
        if init_guess is not None:
            self.init_guess = init_guess

        if init_guess is None:
            raise Exception("Initial guess needs to be provided")

        coefficients = self.init_guess.copy()
        self.coefficients = coefficients.copy()
        rmse_prev = np.inf
        residual = self._calculate_residual(init_guess)
        rmse_best = np.sqrt(np.sum(residual ** 2))
        logger.info(f"RMSE with init guess {rmse_best}")
        for k in range(self.max_iter):
            residual = self._calculate_residual(coefficients)
            jacobian = self._calculate_jacobian(coefficients, step=self.epsilon)
            delta = self.update_step_size * self._calculate_pseudoinverse(jacobian) @ residual
            coefficients = coefficients - delta
            rmse = np.sqrt(np.sum(residual ** 2))
            logger.info(f"Round {k}: RMSE {rmse}")

            if rmse < rmse_best:
                rmse_best = rmse
                self.coefficients = coefficients.copy()

            diff = rmse_prev - rmse
            if diff < self.tolerance_difference and k >= self.min_iter:
                logger.info("RMSE difference between iterations smaller than tolerance. Fit terminated.")
                return self.coefficients
            if rmse < self.tolerance and k >= self.min_iter:
                logger.info("RMSE error smaller than tolerance. Fit terminated.")
                return self.coefficients
            rmse_prev = rmse
        logger.info("Max number of iterations reached. Fit didn't converge.")

        return self.coefficients

    def predict(self, x: np.ndarray):
        """
        Predict response for given x based on fitted coefficients.

        :param x: Independent variable.
        :return: Response vector.
        """
        return self.fit_function(x, self.coefficients)

    def get_residual(self) -> np.ndarray:
        """
        Get residual after fit.

        :return: Residual (y_fitted - y).
        """
        return self._calculate_residual(self.coefficients)

    def get_estimate(self) -> np.ndarray:
        """
        Get estimated response vector based on fit.
        :return: Response vector
        """
        return self.fit_function(self.x, self.coefficients)

    def _calculate_residual(self, coefficients: np.ndarray) -> np.ndarray:
        y_fit = self.fit_function(self.x, coefficients)
        return self.residual_function(y_fit, self.y)

    def _calculate_jacobian(self,
                            x0: np.ndarray,
                            step: float) -> np.ndarray:
        """
        Calculate Jacobian matrix numerically.
        J_ij = d(r_i)/d(x_j)
        """
        y0 = self._calculate_residual(x0)

        jacobian = []
        for i, parameter in enumerate(x0):
            x = x0.copy()
            x[i] += step
            y = self._calculate_residual(x)
            derivative = (y - y0) / step
            jacobian.append(derivative)
        jacobian = np.array(jacobian).T

        return jacobian

    @staticmethod
    def _calculate_pseudoinverse(x: np.ndarray) -> np.ndarray:
        """
        Moore-Penrose inverse.
        """
        return pinv(x.T @ x) @ x.T

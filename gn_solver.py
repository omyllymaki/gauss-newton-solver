import logging
from typing import Callable

import numpy as np
from numpy.linalg import pinv

logger = logging.getLogger(__name__)


class GNSolver:
    """
    Gauss-Newton solver.

    Given response vector y, dependent variable x and fit function f, 
    Minimize sqrt(sum(residual^2)) where residual = f(x, coefficients) - y.
    """

    def __init__(self,
                 fit_function: Callable,
                 max_iter: int = 1000,
                 tolerance_difference: float = 10 ** (-16),
                 tolerance: float = 10 ** (-9),
                 init_guess: np.ndarray = None,
                 ):
        """
        :param fit_function: Function that needs be fitted; y_estimate = fit_function(x, coefficients)
        :param max_iter: Maximum number of iterations for optimization
        :param tolerance_difference: Terminate if RMSE difference between iterations smaller than tolerance
        :param tolerance: Terminate if RMSE is smaller than tolerance
        :param init_guess: Initial guess for coefficients.
        """
        self.fit_function = fit_function
        self.max_iter = max_iter
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

        :param x: Independent variable
        :param y: Response vector
        :param init_guess: Initial guess for coefficients
        :return: Fitted coefficients
        """

        self.x = x
        self.y = y
        if init_guess is not None:
            self.init_guess = init_guess

        if init_guess is None:
            raise Exception("Initial guess needs to be provided")

        self.coefficients = self.init_guess
        rmse_prev = np.inf
        for k in range(self.max_iter):
            residual = self.get_residual()
            jacobian = self._calculate_jacobian(self.coefficients, step=10 ** (-6))
            self.coefficients = self.coefficients - self._calculate_pseudoinverse(jacobian) @ residual
            rmse = np.sqrt(np.sum(residual ** 2))
            logger.info(f"Round {k}: RMSE {rmse}")
            if self.tolerance_difference is not None:
                diff = np.abs(rmse_prev - rmse)
                if diff < self.tolerance_difference:
                    logger.info("RMSE difference between iterations smaller than tolerance. Fit terminated.")
                    return self.coefficients
            if rmse < self.tolerance:
                logger.info("RMSE error smaller than tolerance. Fit terminated.")
                return self.coefficients
            rmse_prev = rmse
        logger.info("Max number of iterations reached. Fit didn't converge.")

        return self.coefficients

    def predict(self, x: np.ndarray):
        """
        Predict response for given x based on fitted coefficients.

        :param x: Independent variable
        :return: Response vector
        """
        return self.fit_function(x, self.coefficients)

    def get_residual(self) -> np.ndarray:
        """
        Get residual after fit.

        :return: Residual, y_fitted - y
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
        return y_fit - self.y

    def _calculate_jacobian(self,
                            x0: np.ndarray,
                            step: float = 10 ** (-6)) -> np.ndarray:
        """
        Calculate Jacobian matrix numerically.
        """
        y0 = self._calculate_residual(x0)

        partial_derivatives = []
        for i, parameter in enumerate(x0):
            x = x0.copy()
            x[i] += step
            y = self._calculate_residual(x)
            derivative = (y - y0) / step
            partial_derivatives.append(derivative)
        partial_derivatives = np.array(partial_derivatives).T

        return partial_derivatives

    @staticmethod
    def _calculate_pseudoinverse(x: np.ndarray) -> np.ndarray:
        """
        Moore-Penrose inverse.
        """
        return pinv(x.T @ x) @ x.T

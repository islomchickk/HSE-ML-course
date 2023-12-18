from __future__ import annotations

from typing import List

import numpy as np

from descents import BaseDescent
from descents import get_descent


class LinearRegression:
    """
    Linear regression class
    """

    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300):
        """
        :param descent_config: gradient descent config
        :param tolerance: stopping criterion for square of euclidean norm of weight difference (float)
        :param max_iter: stopping criterion for iterations (int)
        """
        self.descent: BaseDescent = get_descent(descent_config)

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter

        self.loss_history: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Fitting descent weights for x and y dataset
        :param x: features array
        :param y: targets array
        :return: self
        """
        # TODO: fit weights to x and y
        self.loss_history += [self.descent.calc_loss(x, y)]
        zero_step_weights = self.descent.step(x, y)
        delta = (np.linalg.norm(zero_step_weights, ord=2))**2
        count_nan = np.sum(np.isnan(zero_step_weights))
        self.max_iter -= 1
        while self.max_iter > 0 and count_nan == 0 and delta > self.tolerance:
            self.loss_history += [self.descent.calc_loss(x, y)]
            step_weights = self.descent.step(x, y)
            count_nan = np.sum(np.isnan(zero_step_weights))
            delta = (np.linalg.norm(step_weights, ord=2)) ** 2
            self.max_iter -= 1
        self.loss_history += [self.descent.calc_loss(x, y)]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicting targets for x dataset
        :param x: features array
        :return: prediction: np.ndarray
        """
        return self.descent.predict(x)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculating loss for x and y dataset
        :param x: features array
        :param y: targets array
        """
        return self.descent.calc_loss(x, y)


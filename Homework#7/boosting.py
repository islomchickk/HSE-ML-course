from __future__ import annotations

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        n = x.shape[0]
        boot_n = int(self.subsample * n)
        inds = np.random.choice(range(n), boot_n)

        s = -self.loss_derivative(y[inds], predictions[inds])

        new_model = self.base_model_class()
        new_model.set_params(**self.base_model_params)
        new_model.fit(x[inds], s)

        best_gamma = self.find_optimal_gamma(y, predictions, new_model.predict(x))

        self.gammas.append(best_gamma)
        self.models.append(new_model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        train_loss = [self.loss_fn(y_train, train_predictions)]
        valid_loss = [self.loss_fn(y_valid, valid_predictions)]

        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_train)
            valid_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_valid)

            train_loss.append(self.loss_fn(y_train, train_predictions))
            valid_loss.append(self.loss_fn(y_valid, valid_predictions))

            if self.early_stopping_rounds is not None:
                if len(valid_loss) > self.early_stopping_rounds and \
                        valid_loss[-self.early_stopping_rounds] < valid_loss[-1]:
                    break

        if self.plot:
            sns.lineplot(x=range(len(valid_loss)), y=train_loss, label="train")
            sns.lineplot(x=range(len(valid_loss)), y=valid_loss, label="validation")

    def predict_proba(self, x):
        preds = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            preds += self.learning_rate * gamma * model.predict(x)

        proba = np.zeros((x.shape[0], 2))
        proba[:, 1] = self.sigmoid(preds)
        proba[:, 0] = 1 - proba[:, 1]
        return proba

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        result = np.zeros(self.models[0].feature_importances_.shape[0])
        for model in self.models:
            result += model.feature_importances_

        result /= len(self.models)
        return result

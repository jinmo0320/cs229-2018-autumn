import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    # *** START CODE HERE ***
    clf = GDA()
    clf.fit(x_train, y_train)
    clf.predict(x_eval)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n + 1)

        count_1 = np.sum(y == 1)
        count_0 = np.sum(y == 0)

        phi = count_1 / m
        mu_0 = np.sum(x[y == 0], axis=0) / count_0
        mu_1 = np.sum(x[y == 1], axis=0) / count_1
        delta = x - np.where((y == 0)[:, None], mu_0, mu_1)
        sigma = delta.T @ delta / m
        self.theta[1:] = np.linalg.solve(sigma, mu_1 - mu_0)
        self.theta[0] = (mu_0 @ np.linalg.solve(sigma, mu_0) -
                         mu_1 @ np.linalg.solve(sigma, mu_1)) / 2 + np.log(1 / (1 - phi) - 1)

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-(x @ self.theta)))
        # *** END CODE HERE

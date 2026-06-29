import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train, t_train = util.load_dataset(train_path, 't', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, 't', add_intercept=True)
    model = LogisticRegression()
    model.fit(x_train, t_train)
    util.plot(x_train, t_train, model.theta,
              f'output/p02c_train.png')

    t_pred = model.predict(x_test)
    np.savetxt(pred_path_c, t_pred > 0.5, fmt='%d')
    util.plot(x_test, t_test, model.theta,
              f'output/p02c_test.png')

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    x_train, y_train = util.load_dataset(train_path, 'y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, 'y', add_intercept=True)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    util.plot(x_train, y_train, model.theta,
              f'output/p02d_train.png')

    y_pred = model.predict(x_test)
    np.savetxt(pred_path_d, y_pred > 0.5, fmt='%d')
    util.plot(x_test, y_test, model.theta,
              f'output/p02d_test.png')

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    x_valid, y_valid = util.load_dataset(valid_path, 'y', add_intercept=True)
    labeled_x_valid = x_valid[y_valid == 1]
    correction_factor = model.predict(labeled_x_valid).mean()
    corrected_t_pred = y_pred / correction_factor
    corrected_theta = model.theta + \
        np.log(2 / correction_factor - 1) * np.array([1, 0, 0])

    np.savetxt(pred_path_e, corrected_t_pred > 0.5, fmt='%d')
    util.plot(x_test, y_test, corrected_theta,
              f'output/p02e_pred.png')

    # *** END CODER HERE

# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import matplotlib.pyplot as plt


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    # reg_lambda = 0.0001
    # grad = grad + 2 * reg_lambda * theta

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10  # or 0.1, 1, 100, ...

    # rng = np.random.default_rng(seed=42)
    # noise = rng.normal(loc=0.0, scale=0.1, size=X.shape)
    # noise[:, 0] = 0
    # X += noise

    i = 0
    while True:
        i += 1
        prev_theta = theta

        grad = calc_grad(X, Y, theta)

        # learning_rate /= i**2
        theta = theta - learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
            # print('Training loss = %f' %
            #       np.mean(np.log(1 + np.exp(-Y * X.dot(theta)))))
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return


def main():
    # Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=False)
    # plt.figure()
    # util.plot_points(Xa, (Ya == 1).astype(int))
    # plt.savefig('output/ds1_a.png')

    # Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=False)
    # plt.figure()
    # util.plot_points(Xb, (Yb == 1).astype(int))
    # plt.savefig('output/ds1_b.png')

    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb)


if __name__ == '__main__':
    main()

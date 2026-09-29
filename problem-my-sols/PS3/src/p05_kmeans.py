from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

K = 16


def main():
    # load data
    image = imread('../data/peppers-large.tiff')
    small_image = imread('../data/peppers-small.tiff')
    plt.imsave('./output/large_original.png', image)
    plt.imsave('./output/small_original.png', small_image)
    x = small_image.reshape(-1, 3)

    # init
    rng = np.random.default_rng(seed=39)
    indices = rng.choice(len(x), size=K, replace=False)
    centroid = x[indices]

    # k-means
    centroid = run_k_means(x, centroid)

    # testing
    w, h, _ = image.shape
    x = image.reshape(-1, 3)
    x_compressed = finding_closest_cluster(x, centroid)
    image_compressed = x_compressed.reshape(w, h, 3)
    plt.imsave('./output/large_compressed.png',
               np.clip(image_compressed, 0, 255).astype(np.uint8))


def run_k_means(x, centroid):
    eps = 1e-3
    max_iter = 30

    it = 0
    j = prev_j = None
    while it < max_iter and (prev_j is None or np.abs(j - prev_j) >= eps):
        norm = np.sum((x[:, None, :] - centroid[None, :, :]) ** 2, axis=2)
        closest_cluster = np.argmin(norm, axis=1)
        one_hot = np.eye(K, dtype=int)[closest_cluster]
        centroid = (one_hot.T @ x) / \
            one_hot.sum(axis=0).reshape(-1, 1)

        # Compute the cost of the data to check for convergence.
        it += 1
        prev_j = j
        j = np.sum((x - centroid[closest_cluster]) ** 2)

    print(f"iter: {it}")
    return centroid


def finding_closest_cluster(x, centroid):
    norm = np.sum((x[:, None, :] - centroid[None, :, :]) ** 2, axis=2)
    closest_cluster = np.argmin(norm, axis=1)
    return centroid[closest_cluster]


if __name__ == '__main__':
    main()

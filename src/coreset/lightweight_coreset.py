import numpy as np


class LightweightCoreset:

    def __init__(self, coreset_size=500):

        self.coreset_size = coreset_size

    def fit_transform(self, X):

        N = X.shape[0]

        # dataset mean
        mean = np.mean(X, axis=0)

        # squared distance from mean
        dist_sq = np.sum((X - mean) ** 2, axis=1)

        total_dist = np.sum(dist_sq)

        uniform_part = 1 / (2 * N)

        sensitivity = (
            uniform_part
            + dist_sq / (2 * total_dist)
        )

        probabilities = sensitivity / np.sum(sensitivity)

        indices = np.random.choice(
            N,
            size=self.coreset_size,
            replace=False,
            p=probabilities
        )

        coreset_points = X[indices]

        weights = (
            1 /
            (self.coreset_size * probabilities[indices])
        )

        return (
            coreset_points,
            weights,
            indices
        )
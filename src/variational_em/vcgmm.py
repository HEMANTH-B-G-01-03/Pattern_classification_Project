import numpy as np


def estimate_initial_variance(
    X,
    centers
):

    distances = np.min(
        [
            np.sum(
                (X - center) ** 2,
                axis=1
            )
            for center in centers
        ],
        axis=0
    )

    variance = (
        np.mean(distances)
        / X.shape[1]
    )

    return variance
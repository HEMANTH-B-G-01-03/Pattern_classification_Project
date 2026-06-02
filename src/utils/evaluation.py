import numpy as np


def compute_distortion(
    X,
    centers
):
    """
    Sum of squared distances
    to nearest cluster center
    """

    distances = []

    for point in X:

        d = np.min(
            np.sum(
                (centers - point) ** 2,
                axis=1
            )
        )

        distances.append(d)

    return np.sum(
        distances
    )
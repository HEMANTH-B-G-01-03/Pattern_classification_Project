import numpy as np


class NeighborhoodSearch:

    def __init__(
        self,
        n_neighbors=5
    ):
        self.n_neighbors = n_neighbors

    def build(
        self,
        centers
    ):

        n_clusters = centers.shape[0]

        Gc = []

        for i in range(
            n_clusters
        ):

            distances = np.sum(
                (
                    centers - centers[i]
                ) ** 2,
                axis=1
            )

            neighbors = np.argsort(
                distances
            )[
                :self.n_neighbors
            ]

            Gc.append(
                neighbors
            )

        return Gc
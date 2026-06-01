import numpy as np


class AFKMC2:
    def __init__(
        self,
        n_clusters=20,
        random_state=42
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def initialize(self, X):

        np.random.seed(self.random_state)

        n_samples = X.shape[0]

        first_idx = np.random.randint(
            n_samples
        )

        centers = [X[first_idx]]

        for _ in range(
            1,
            self.n_clusters
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

            probabilities = (
                distances /
                np.sum(distances)
            )

            next_idx = np.random.choice(
                n_samples,
                p=probabilities
            )

            centers.append(
                X[next_idx]
            )

        return np.array(centers)
import numpy as np


class MStep:

    def run(
        self,
        X,
        K_sets,
        responsibilities,
        n_clusters
    ):

        n_features = X.shape[1]

        new_centers = np.zeros(
            (n_clusters, n_features)
        )

        denominator = np.zeros(
            n_clusters
        )

        for i in range(len(X)):

            point = X[i]

            clusters = K_sets[i]

            probs = responsibilities[i]

            for idx, cluster_id in enumerate(
                clusters
            ):

                resp = probs[idx]

                new_centers[
                    cluster_id
                ] += resp * point

                denominator[
                    cluster_id
                ] += resp

        for c in range(n_clusters):

            if denominator[c] > 0:

                new_centers[c] /= (
                    denominator[c]
                )

        return new_centers
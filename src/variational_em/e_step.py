# import numpy as np


# class VariationalEStep:

#     def __init__(
#         self,
#         n_candidates=5
#     ):
#         self.n_candidates = n_candidates

#     def run(
#         self,
#         X,
#         centers,
#         sigma2
#     ):

#         n_samples = X.shape[0]

#         K_sets = []
#         responsibilities = []

#         for i in range(n_samples):

#             point = X[i]

#             distances = np.sum(
#                 (centers - point) ** 2,
#                 axis=1
#             )

#             nearest_idx = np.argsort(
#                 distances
#             )[:self.n_candidates]

#             K_sets.append(
#                 nearest_idx
#             )

#             selected_distances = (
#                 distances[nearest_idx]
#             )

#             probs = np.exp(
#                 -selected_distances
#                 /
#                 (2 * sigma2)
#             )

#             probs = probs / np.sum(
#                 probs
#             )

#             responsibilities.append(
#                 probs
#             )

#         return (
#             K_sets,
#             responsibilities
#         )



import numpy as np


class VariationalEStep:

    def __init__(
        self,
        n_candidates=5
    ):
        self.n_candidates = n_candidates

    def run(
        self,
        X,
        centers,
        sigma2
    ):

        n_samples = X.shape[0]

        K_sets = []
        responsibilities = []

        for i in range(n_samples):

            point = X[i]

            distances = np.sum(
                (centers - point) ** 2,
                axis=1
            )

            nearest_idx = np.argsort(
                distances
            )[:self.n_candidates]

            K_sets.append(
                nearest_idx
            )

            selected_distances = (
                distances[nearest_idx]
            )

            # Stable softmax
            scores = (
                -selected_distances
                /
                (2 * sigma2)
            )

            scores = scores - np.max(
                scores
            )

            probs = np.exp(
                scores
            )

            probs = probs / np.sum(
                probs
            )

            responsibilities.append(
                probs
            )

        return (
            K_sets,
            responsibilities
        )
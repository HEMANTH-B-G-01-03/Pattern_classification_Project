# # import numpy as np


# # def estimate_initial_variance(
# #     X,
# #     centers
# # ):

# #     distances = np.min(
# #         [
# #             np.sum(
# #                 (X - center) ** 2,
# #                 axis=1
# #             )
# #             for center in centers
# #         ],
# #         axis=0
# #     )

# #     variance = (
# #         np.mean(distances)
# #         / X.shape[1]
# #     )

# #     return variance

# import numpy as np

# from src.variational_em.e_step import (
#     VariationalEStep
# )

# from src.variational_em.m_step import (
#     MStep
# )


# def estimate_initial_variance(
#     X,
#     centers
# ):

#     distances = np.min(
#         [
#             np.sum(
#                 (X - center) ** 2,
#                 axis=1
#             )
#             for center in centers
#         ],
#         axis=0
#     )

#     variance = (
#         np.mean(distances)
#         / X.shape[1]
#     )

#     return variance


# class VCGMM:

#     def __init__(
#         self,
#         n_clusters=20,
#         n_candidates=5,
#         max_iter=20,
#         tol=1e-4
#     ):

#         self.n_clusters = n_clusters
#         self.n_candidates = n_candidates
#         self.max_iter = max_iter
#         self.tol = tol

#     def fit(
    
#     def fit(
#     self,
#     X,
#     centers,
#     sigma2
# ):

#     e_step = VariationalEStep(
#         self.n_candidates
#     )

#     m_step = MStep()

#     previous_centers = centers.copy()

#     self.movement_history = []

#     for iteration in range(
#         self.max_iter
#     ):


import numpy as np

from src.variational_em.e_step import (
    VariationalEStep
)

from src.variational_em.m_step import (
    MStep
)


def estimate_initial_variance(
    X,
    centers
):
    """
    Estimate initial variance sigma²
    """

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


class VCGMM:

    def __init__(
        self,
        n_clusters=20,
        n_candidates=5,
        max_iter=20,
        tol=1e-4
    ):

        self.n_clusters = n_clusters
        self.n_candidates = n_candidates
        self.max_iter = max_iter
        self.tol = tol

        self.centers = None
        self.movement_history = []

    def fit(
        self,
        X,
        centers,
        sigma2
    ):

        e_step = VariationalEStep(
            self.n_candidates
        )

        m_step = MStep()

        previous_centers = centers.copy()

        self.movement_history = []

        for iteration in range(
            self.max_iter
        ):

            K_sets, responsibilities = (
                e_step.run(
                    X,
                    centers,
                    sigma2
                )
            )

            centers = m_step.run(
                X,
                K_sets,
                responsibilities,
                self.n_clusters
            )

            movement = np.linalg.norm(
                centers - previous_centers
            )

            self.movement_history.append(
                movement
            )

            print(
                f"Iteration {iteration + 1} | Movement: {movement:.6f}"
            )

            if movement < self.tol:

                print(
                    "\nConverged!"
                )

                break

            previous_centers = (
                centers.copy()
            )

        self.centers = centers

        return centers

    def get_centers(self):

        return self.centers

    def get_history(self):

        return self.movement_history
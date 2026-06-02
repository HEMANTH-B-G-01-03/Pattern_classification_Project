# # from src.utils.dataset_loader import load_dataset
# # from src.coreset.lightweight_coreset import LightweightCoreset


# # def main():

# #     # Load Dataset
# #     X, y = load_dataset()

# #     # Create LWCS object
# #     lwcs = LightweightCoreset(
# #         coreset_size=300
# #     )

# #     # Generate coreset
# #     X_core, weights, indices = lwcs.fit_transform(X)

# #     print("\nLWCS Output")
# #     print("------------")
# #     print(f"Original Shape : {X.shape}")
# #     print(f"Coreset Shape  : {X_core.shape}")
# #     print(f"Weights Shape  : {weights.shape}")


# # if __name__ == "__main__":
# #     main()



# from src.utils.dataset_loader import load_dataset
# from src.coreset.lightweight_coreset import LightweightCoreset
# from src.seeding.afkmc2 import AFKMC2
# from src.variational_em.vcgmm import (
#     estimate_initial_variance
# )
# from src.variational_em.e_step import (
#     VariationalEStep
# )

# from src.variational_em.m_step import (
#     MStep
# )
# from src.variational_em.vcgmm import (
#     VCGMM
# )

# from src.utils.visualization import (
#     plot_convergence
# )

# from src.variational_em.neighborhood import (
#     NeighborhoodSearch
# )

# def main():

#     X, y = load_dataset()

#     lwcs = LightweightCoreset(
#         coreset_size=1000
#     )

#     X_core, weights, indices = (
#         lwcs.fit_transform(X)
#     )

#     print("\nLWCS Output")
#     print("------------")
#     print(
#         f"Original Shape : {X.shape}"
#     )
#     print(
#         f"Coreset Shape  : {X_core.shape}"
#     )

#     afkmc2 = AFKMC2(
#         n_clusters=50
        
        
        
#     )

#     centers = afkmc2.initialize(
#         X_core
#     )

#     print("\nAFK-MC2 Output")
#     print("--------------")
#     print(
#         f"Centers Shape : {centers.shape}"
#     )
#     variance = estimate_initial_variance(
#     X_core,
#     centers
# )
    

#     print("\nInitial Variance")
#     print("----------------")
#     print(
#         f"Sigma² : {variance:.6f}"
#     )
    
    
    
#     neighborhood = (
#     NeighborhoodSearch(
#         n_neighbors=5
#     )
# )

#     Gc = neighborhood.build(
#         centers
#     )

#     print("\nNeighborhood Sets")
#     print("------------------")
#     print(
#         f"Clusters : {len(Gc)}"
#     )

#     print(
#         "First Neighborhood:"
#     )

#     print(
#         Gc[0]
#     )

#     e_step = VariationalEStep(
#         n_candidates=5
#     )

#     K_sets, responsibilities = (
#         e_step.run(
#             X_core,
#             centers,
#             variance
#         )
#     )

#     print("\nVariational E-Step")
#     print("------------------")
#     print(
#         f"K Sets      : {len(K_sets)}"
#     )
#     print(
#         f"Responsibilities : {len(responsibilities)}"
#     )
    

#     print("\nFirst K Set")
#     print(K_sets[0])

#     print("\nFirst Responsibility")
#     print(responsibilities[0])
    
        
#     m_step = MStep()

#     new_centers = m_step.run(
#         X_core,
#         K_sets,
#         responsibilities,
#         centers.shape[0]
#     )

#     print("\nM-Step")
#     print("------")
#     print(
#         f"Updated Centers Shape : {new_centers.shape}"
#     )
    
#     print("\nTraining vc-GMM")
#     print("----------------")

#     model = VCGMM(
#         n_clusters=50,
#         n_candidates=5,
#         max_iter=10
#     )

#     final_centers = model.fit(
#         X_core,
#         centers
#         variance
#     )

#     print("\nFinal Centers")
#     print(
#         final_centers.shape
#     )
#     plot_convergence(
#     model.movement_history
#     )

# if __name__ == "__main__":
#     main()


import time

from src.utils.dataset_loader import load_dataset
from src.coreset.lightweight_coreset import LightweightCoreset
from src.seeding.afkmc2 import AFKMC2

from src.variational_em.vcgmm import (
    estimate_initial_variance,
    VCGMM
)

from src.variational_em.neighborhood import (
    NeighborhoodSearch
)

from src.utils.visualization import (
    plot_convergence
)
from src.utils.evaluation import (
    compute_distortion
)


def main():

    # ==================================================
    # DATASET
    # ==================================================

    X, y = load_dataset()

    # ==================================================
    # LIGHTWEIGHT CORESET
    # ==================================================

    lwcs = LightweightCoreset(
        coreset_size=1000
    )

    lwcs_start = time.time()

    X_core, weights, indices = (
        lwcs.fit_transform(X)
    )

    lwcs_time = (
        time.time() - lwcs_start
    )

    print("\nLWCS Output")
    print("------------")
    print(f"Original Shape : {X.shape}")
    print(f"Coreset Shape  : {X_core.shape}")

    # ==================================================
    # AFK-MC² SEEDING
    # ==================================================

    afkmc2 = AFKMC2(
        n_clusters=50
    )

    seed_start = time.time()

    centers = afkmc2.initialize(
        X_core
    )

    seed_time = (
        time.time() - seed_start
    )

    print("\nAFK-MC2 Output")
    print("--------------")
    print(f"Centers Shape : {centers.shape}")

    # ==================================================
    # INITIAL VARIANCE
    # ==================================================

    variance = estimate_initial_variance(
        X_core,
        centers
    )

    print("\nInitial Variance")
    print("----------------")
    print(
        f"Sigma² : {variance:.6f}"
    )

    # ==================================================
    # NEIGHBORHOOD SETS
    # ==================================================

    neighborhood = NeighborhoodSearch(
        n_neighbors=5
    )

    Gc = neighborhood.build(
        centers
    )

    print("\nNeighborhood Sets")
    print("------------------")
    print(f"Clusters : {len(Gc)}")
    print("First Neighborhood:")
    print(Gc[0])

    # ==================================================
    # VC-GMM TRAINING
    # ==================================================

    print("\nTraining vc-GMM")
    print("----------------")

    model = VCGMM(
        n_clusters=50,
        n_candidates=5,
        max_iter=10
    )

    train_start = time.time()

    final_centers = model.fit(
        X_core,
        centers,
        variance
    )

    train_time = (
        time.time() - train_start
    )

    print("\nFinal Centers")
    print(final_centers.shape)
    
    
    
    distortion = compute_distortion(
    X_core,
    final_centers
)

    print("\nClustering Objective")
    print("--------------------")
    print(
        f"Distortion : {distortion:.2f}"
    )

    # ==================================================
    # RUNTIME STATISTICS
    # ==================================================

    total_time = (
        lwcs_time +
        seed_time +
        train_time
    )

    print("\nRuntime Statistics")
    print("------------------")

    print(
        f"LWCS Time      : {lwcs_time:.4f} sec"
    )

    print(
        f"Seeding Time   : {seed_time:.4f} sec"
    )

    print(
        f"vc-GMM Time    : {train_time:.4f} sec"
    )

    print(
        f"Total Time     : {total_time:.4f} sec"
    )

    # ==================================================
    # CONVERGENCE PLOT
    # ==================================================

    plot_convergence(
        model.movement_history
    )


if __name__ == "__main__":
    main()
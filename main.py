# from src.utils.dataset_loader import load_dataset
# from src.coreset.lightweight_coreset import LightweightCoreset


# def main():

#     # Load Dataset
#     X, y = load_dataset()

#     # Create LWCS object
#     lwcs = LightweightCoreset(
#         coreset_size=300
#     )

#     # Generate coreset
#     X_core, weights, indices = lwcs.fit_transform(X)

#     print("\nLWCS Output")
#     print("------------")
#     print(f"Original Shape : {X.shape}")
#     print(f"Coreset Shape  : {X_core.shape}")
#     print(f"Weights Shape  : {weights.shape}")


# if __name__ == "__main__":
#     main()



from src.utils.dataset_loader import load_dataset
from src.coreset.lightweight_coreset import LightweightCoreset
from src.seeding.afkmc2 import AFKMC2
from src.variational_em.vcgmm import (
    estimate_initial_variance
)
from src.variational_em.e_step import (
    VariationalEStep
)

from src.variational_em.m_step import (
    MStep
)
from src.variational_em.vcgmm import (
    VCGMM
)

from src.utils.visualization import (
    plot_convergence
)

from src.variational_em.neighborhood import (
    NeighborhoodSearch
)

def main():

    X, y = load_dataset()

    lwcs = LightweightCoreset(
        coreset_size=1000
    )

    X_core, weights, indices = (
        lwcs.fit_transform(X)
    )

    print("\nLWCS Output")
    print("------------")
    print(
        f"Original Shape : {X.shape}"
    )
    print(
        f"Coreset Shape  : {X_core.shape}"
    )

    afkmc2 = AFKMC2(
        n_clusters=50
        
        
        
    )

    centers = afkmc2.initialize(
        X_core
    )

    print("\nAFK-MC2 Output")
    print("--------------")
    print(
        f"Centers Shape : {centers.shape}"
    )
    variance = estimate_initial_variance(
    X_core,
    centers
)
    

    print("\nInitial Variance")
    print("----------------")
    print(
        f"Sigma² : {variance:.6f}"
    )
    
    
    
    neighborhood = (
    NeighborhoodSearch(
        n_neighbors=5
    )
)

    Gc = neighborhood.build(
        centers
    )

    print("\nNeighborhood Sets")
    print("------------------")
    print(
        f"Clusters : {len(Gc)}"
    )

    print(
        "First Neighborhood:"
    )

    print(
        Gc[0]
    )

    e_step = VariationalEStep(
        n_candidates=5
    )

    K_sets, responsibilities = (
        e_step.run(
            X_core,
            centers,
            variance
        )
    )

    print("\nVariational E-Step")
    print("------------------")
    print(
        f"K Sets      : {len(K_sets)}"
    )
    print(
        f"Responsibilities : {len(responsibilities)}"
    )
    

    print("\nFirst K Set")
    print(K_sets[0])

    print("\nFirst Responsibility")
    print(responsibilities[0])
    
        
    m_step = MStep()

    new_centers = m_step.run(
        X_core,
        K_sets,
        responsibilities,
        centers.shape[0]
    )

    print("\nM-Step")
    print("------")
    print(
        f"Updated Centers Shape : {new_centers.shape}"
    )
    
    print("\nTraining vc-GMM")
    print("----------------")

    model = VCGMM(
        n_clusters=50,
        n_candidates=5,
        max_iter=10
    )

    final_centers = model.fit(
        X_core,
        centers,
        variance
    )

    print("\nFinal Centers")
    print(
        final_centers.shape
    )
    plot_convergence(
    model.movement_history
    )

if __name__ == "__main__":
    main()
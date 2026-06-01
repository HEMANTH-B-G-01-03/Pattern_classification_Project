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


def main():

    X, y = load_dataset()

    lwcs = LightweightCoreset(
        coreset_size=300
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
        n_clusters=20
    )

    centers = afkmc2.initialize(
        X_core
    )

    print("\nAFK-MC2 Output")
    print("--------------")
    print(
        f"Centers Shape : {centers.shape}"
    )


if __name__ == "__main__":
    main()
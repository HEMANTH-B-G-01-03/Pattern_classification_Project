from src.utils.dataset_loader import load_dataset
from src.coreset.lightweight_coreset import LightweightCoreset


def main():

    # Load Dataset
    X, y = load_dataset()

    # Create LWCS object
    lwcs = LightweightCoreset(
        coreset_size=300
    )

    # Generate coreset
    X_core, weights, indices = lwcs.fit_transform(X)

    print("\nLWCS Output")
    print("------------")
    print(f"Original Shape : {X.shape}")
    print(f"Coreset Shape  : {X_core.shape}")
    print(f"Weights Shape  : {weights.shape}")


if __name__ == "__main__":
    main()
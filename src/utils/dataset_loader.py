from sklearn.datasets import load_digits


def load_dataset():
    """
    Development dataset.

    Later:
    - CIFAR-10
    - KDD
    - SUSY

    For now:
    - Digits dataset
    """

    X, y = load_digits(return_X_y=True)

    print("\nDataset Loaded Successfully")
    print(f"Samples : {X.shape[0]}")
    print(f"Features: {X.shape[1]}")

    return X, y
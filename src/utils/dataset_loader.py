# from sklearn.datasets import load_digits


# def load_dataset():
#     """
#     Development dataset.

#     Later:
#     - CIFAR-10
#     - KDD
#     - SUSY

#     For now:
#     - Digits dataset
#     """

#     X, y = load_digits(return_X_y=True)

#     print("\nDataset Loaded Successfully")
#     print(f"Samples : {X.shape[0]}")
#     print(f"Features: {X.shape[1]}")

#     return X, y


from torchvision.datasets import CIFAR10
import numpy as np


def load_dataset():

    dataset = CIFAR10(
        root="./data",
        train=True,
        download=True
    )

    X = dataset.data

    y = np.array(
        dataset.targets
    )

    X = X.reshape(
        len(X),
        -1
    )

    X = X.astype(
        np.float32
    )

    print(
        "\nCIFAR-10 Loaded Successfully"
    )

    print(
        f"Samples : {X.shape[0]}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    return X, y
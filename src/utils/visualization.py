import matplotlib.pyplot as plt


def plot_convergence(history):

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(history) + 1),
        history,
        marker="o"
    )

    plt.title(
        "vc-GMM Convergence"
    )

    plt.xlabel(
        "Iteration"
    )

    plt.ylabel(
        "Center Movement"
    )

    plt.grid(True)

    plt.savefig(
        "results/convergence_plot.png"
    )

    plt.show()
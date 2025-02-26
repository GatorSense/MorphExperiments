import matplotlib.pyplot as plt

def plot_heatmap(data):
    plt.clf()
    heatmap = plt.imshow(data.T, cmap='viridis')
    plt.colorbar()

    plt.xlabel("True label")
    plt.ylabel("Predicted Label")
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.yticks(ticks=[0, 1], labels=["Not Three", "Three"])
    plt.title('Heatmap')

    for i in range(data.T.shape[0]):
        for j in range(data.T.shape[1]):
            plt.text(j, i, f"{data[j, i]:.0f}", ha='center', va='center', color='w')

    return heatmap.figure

def plot_hit_filters(selected_3):
    fig, axes = plt.subplots(2, 5, figsize=(8,4))

    for i in range(10):
        image = selected_3[i][0][0]
        if (i < 5):
            ax = axes[0][i]
        else:
            ax = axes[1][i-5]

        ax.imshow(image, cmap="gray")
        ax.set_title(f"Filter {i + 1}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle ("Hit Filters")
    fig.tight_layout()
    # plt.show()

    return fig, plt
    # plt.savefig("filters/initialize/initial_filters_hit.png")
    # experiment.log_figure(figure_name="filters_hit", figure=fig)

def plot_miss_filters(selected_3):
    fig, axes = plt.subplots(2, 5, figsize=(8,4))

    for i in range(10):
        image = selected_3[i][0][0]
        if (i < 5):
            ax = axes[0][i]
        else:
            ax = axes[1][i-5]

        ax.imshow(1-image, cmap="gray")
        ax.set_title(f"Filter {i + 1}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle ("Miss Filters")
    fig.tight_layout()
    # plt.show()
    return fig, plt
    # plt.savefig("filters/initialize/initial_filters_miss.png")
    # experiment.log_figure(figure_name="filters_miss", figure=fig)
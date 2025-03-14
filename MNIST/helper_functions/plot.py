import matplotlib.pyplot as plt

def visualize_filters(layer, dir='filters/', title="Filters"):
    # Extract filter weights
    K_hit = layer.K_hit.data.cpu().numpy()  # Convert to NumPy
    K_miss = layer.K_miss.data.cpu().numpy()
    
    out_channels, in_channels, kernel_size, _ = K_hit.shape

    # Ensure the directory exists
    os.makedirs(dir, exist_ok=True)

    # Iterate over filters
    for i in range(out_channels):
        for j in range(in_channels):
            # Save K_hit
            plt.imshow(K_hit[i, j], cmap='gray', interpolation='nearest')
            plt.title(f"K_hit [{i},{j}]")
            plt.axis('off')
            plt.savefig(os.path.join(dir, f"filter_{i}_hit.png"))
            plt.clf()

            # Save K_miss
            plt.imshow(K_miss[i, j], cmap='gray', interpolation='nearest')
            plt.title(f"K_miss [{i},{j}]")
            plt.axis('off')
            plt.savefig(os.path.join(dir, f"filter_{i}_miss.png"))
            plt.clf()

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
    plt.clf()
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
    plt.clf()
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

def plot_filters(filter_layer):
    plt.clf()
    filter = filter_layer.data.cpu().numpy()
    
    out_channels, in_channels, kernel_size, _ = filter.shape

    fig, axes = plt.subplots(2, 5, figsize=(16,8))

    for i in range(out_channels):
        for j in range(in_channels):
            if (i < 5):
                ax_hit = axes[0][i]
            else:
                ax_hit = axes[1][i-5]
            
            ax_hit.imshow(filter[i, j], cmap='gray', interpolation='nearest')
            ax_hit.set_title(f"filter [{i},{j}]")
            ax_hit.set_xticks([])
            ax_hit.set_yticks([])
        
        fig.suptitle("Filters")
        fig.tight_layout()

    return fig, plt

import math
def plot_conv_filters(filter_layer):
    plt.clf()
    filter = filter_layer.weight.data.cpu().numpy()

    out_channels, in_channels, kernel_size, _ = filter.shape

    # Calculate the grid size based on the number of filters (out_channels)
    cols = 5  # Fixed number of columns (you can adjust this)
    rows = math.ceil(out_channels / cols)  # Calculate the number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))

    for i in range(out_channels):
        for j in range(in_channels):
            row = i // cols
            col = i % cols
            
            ax_hit = axes[row][col] if rows > 1 else axes[col]  # Adjust for single row

            ax_hit.imshow(filter[i, j], cmap='gray', interpolation='nearest')
            ax_hit.set_title(f"filter [{i},{j}]")
            ax_hit.set_xticks([])
            ax_hit.set_yticks([])

    fig.suptitle("Filters")
    fig.tight_layout()

    return fig, plt
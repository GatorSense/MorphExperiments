import matplotlib.pyplot as plt
import os
import math
import numpy as np

# Was at the very top of the script, can't remember how it's different from plot_filters
def visualize_filters(layer, dir='filters/', title="Filters"):
    # Extract filter weights
    K_hit = layer.K_hit.data.cpu().numpy()  # Convert to NumPy
    K_miss = layer.K_miss.data.cpu().numpy()
    
    out_channels, in_channels, _, _ = K_hit.shape

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

def plot_heatmap(data, experiment, epoch):
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

    experiment.log_figure(figure_name="heatmap", figure=heatmap.figure, step=epoch)

def plot_filters_initial(selected_3, experiment, filter_name):
    plt.clf()
    os.makedirs("filters/initialize/", exist_ok=True)

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

    fig.suptitle (f"{filter_name} Filters")
    fig.tight_layout()

    plt.savefig(f"filters/initialize/initial_filters_{filter_name}.png")
    experiment.log_figure(figure_name=f"filters_{filter_name}", figure=fig)

    plt.clf()
    plt.close()

# merge with above function in the future
def plot_morphed_filters_initial(filters, experiment, filter_name):
    plt.clf()
    os.makedirs("filters/initialize/", exist_ok=True)

    fig, axes = plt.subplots(2, 5, figsize=(8,4))

    for i, filter in enumerate(filters):
        image = filter[0][0]
        if (i < 5):
            ax = axes[0][i]
        else:
            ax = axes[1][i-5]

        ax.imshow(image, cmap="gray")
        ax.set_title(f"Filter {i + 1}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle (f"{filter_name} Filters")
    fig.tight_layout()

    plt.savefig(f"filters/initialize/initial_filters_{filter_name}.png")
    experiment.log_figure(figure_name=f"filters_{filter_name}", figure=fig)

    plt.clf()
    plt.close()

# Used in forward function
def plot_filters_forward(filter_layer, experiment, epoch, filter_name):
    plt.clf()

    os.makedirs('filters/', exist_ok=True)
    os.makedirs("feature_maps/morph", exist_ok=True)

    filter = filter_layer.data.cpu().numpy()
    out_channels, in_channels, _, _ = filter.shape

    fig, axes = plt.subplots(2, 5, figsize=(16,8))

    for i in range(out_channels):
        for j in range(in_channels):
            if (i < 5):
                ax_hit = axes[0][i]
            else:
                ax_hit = axes[1][i-5]
            
            im = ax_hit.imshow(filter[i, j], cmap='gray', interpolation='nearest')
            fig.colorbar(im, ax=ax_hit)
            ax_hit.set_title(f"filter [{i},{j}]")
            ax_hit.set_xticks([])
            ax_hit.set_yticks([])
        
        fig.suptitle("Filters")
        fig.tight_layout()

    # Locally save the filter images only if it's epoch 100
    if epoch == 100:
        plt.savefig(os.path.join(
            'filters',
            f"{filter_name}_filter_epoch{epoch}.png"))
    experiment.log_figure(figure_name=f'filters/{filter_name}',
                          figure=fig,
                          step=epoch)
    plt.clf()
    plt.close()

def hit_miss_histograms(morph_dict, mode):
    figs = {}

    assert (mode == "Hit" or mode == "Miss")
    
    for key in ["0", "1"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(morph_dict[key], bins=50, alpha=0.75)
        if key == "0":
            ax.set_title(f"{mode} Values for KMNIST Images")
        if key == "1":
            ax.set_title(f"{mode} Values for Three Images")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_xlim(morph_dict[key].min(), morph_dict[key].max())
        figs[key] = fig
    
    return figs

def plot_hit_miss_histogram(morph_dict, mode, experiment, epoch):
    assert (mode == "Hit" or mode == "Miss")

    fm_dict_np = {}          
    for key in morph_dict.keys():
        fm_dict_np[key] = np.concatenate(morph_dict[key]).flatten()

    hists = fm_histograms(fm_dict_np)
    experiment.log_figure(figure_name=f'{mode}/KMNIST Images', figure=hists["0"], step=epoch)
    experiment.log_figure(figure_name=f'{mode}/Three Images', figure=hists["1"], step=epoch)

def fm_histograms(fm_dict):
    figs = {}
    
    for key in ["0", "1"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(fm_dict[key], bins=50, alpha=0.75)
        if key == "0":
            ax.set_title(f"Feature Map Values for KMNIST Images")
        if key == "1":
            ax.set_title(f"Feature Map Values for Three Images")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_xlim(fm_dict[key].min(), fm_dict[key].max())
        figs[key] = fig
    
    return figs

def plot_fm_histogram(fm_dict, experiment, epoch):
    fm_dict_np = {}          
    for key in fm_dict.keys():
        fm_dict_np[key] = np.concatenate(fm_dict[key]).flatten()

    hists = fm_histograms(fm_dict_np)
    experiment.log_figure(figure_name=f'FMs/KMNIST Images', figure=hists["0"], step=epoch)
    experiment.log_figure(figure_name=f'FMs/Three Images', figure=hists["1"], step=epoch)

def fm_histograms_test(fm_dict):
    figs = {}
    
    for key in ["0-2", "4-9"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(fm_dict[key], bins=50, alpha=0.75)
        if key == "0-2":
            ax.set_title(f"Feature Map Values for 0-2 Images")
        if key == "4-9":
            ax.set_title(f"Feature Map Values for 4-9 Images")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_xlim(fm_dict[key].min(), fm_dict[key].max())
        figs[key] = fig
    
    return figs

def plot_fm_histogram_test(fm_dict, experiment, epoch):
    fm_dict_np = {}          
    for key in fm_dict.keys():
        fm_dict_np[key] = np.concatenate(fm_dict[key]).flatten()

    hists = fm_histograms_test(fm_dict_np)
    experiment.log_figure(figure_name=f'test/0-2', figure=hists["0-2"], step=epoch)
    experiment.log_figure(figure_name=f'test/4-9', figure=hists["4-9"], step=epoch)

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
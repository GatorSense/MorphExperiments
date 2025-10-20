import matplotlib.pyplot as plt
import os
import math
import numpy as np

# Was at the very top of the script, can't remember how it's different from plot_filters
def visualize_filters(layer, dir='filters/', title="Filters"):
    K_hit = layer.K_hit.data.cpu().numpy()
    K_miss = layer.K_miss.data.cpu().numpy()
    out_channels, in_channels, _, _ = K_hit.shape

    os.makedirs(dir, exist_ok=True)

    for i in range(out_channels):
        for j in range(in_channels):
            # Save K_hit
            plt.imshow(K_hit[i, j], interpolation='nearest', cmap='viridis')
            plt.title(f"K_hit [{i},{j}]")
            plt.axis('off')
            plt.savefig(os.path.join(dir, f"filter_{i}_hit.png"))
            plt.clf()

            # Save K_miss
            plt.imshow(K_miss[i, j], interpolation='nearest', cmap='viridis')
            plt.title(f"K_miss [{i},{j}]")
            plt.axis('off')
            plt.savefig(os.path.join(dir, f"filter_{i}_miss.png"))
            plt.clf()

def plot_heatmap(data, experiment, epoch):
    plt.clf()
    plt.figure(figsize=(28, 16)) 

    heatmap = plt.imshow(data.T, cmap='viridis')
    plt.colorbar()

    classes = [str(lbl) for lbl in range(1, 32)]
    ticks = [lbl for lbl in range(1, 32)]

    plt.xlabel("True label")
    plt.ylabel("Predicted Label")
    plt.xticks(ticks=ticks, labels=classes, rotation=90)
    plt.yticks(ticks=[0, 1], labels=["Not Target", "Target"])
    plt.title('Heatmap')

    for i in range(data.T.shape[0]):
        for j in range(data.T.shape[1]):
            plt.text(j, i, f"{data[j, i]:.0f}", ha='center', va='center', color='w')

    experiment.log_figure(figure_name="heatmap", figure=plt.gcf(), step=epoch)

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
def plot_morph_filters_forward(filter_layer, experiment, epoch, filter_name):
    plt.clf()

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

def plot_conv_filters_forward(filter_layer, experiment, epoch):
    plt.clf()
    filter = filter_layer.weight.data.cpu().numpy()

    out_channels, in_channels, _, _ = filter.shape

    # Calculate the grid size based on the number of filters (out_channels)
    cols = 5  # Fixed number of columns (you can adjust this)
    rows = math.ceil(out_channels / cols)  # Calculate the number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))

    for i in range(out_channels):
        for j in range(in_channels):
            row = i // cols
            col = i % cols
            
            ax_hit = axes[row][col] if rows > 1 else axes[col]  # Adjust for single row

            im = ax_hit.imshow(filter[i, j], cmap='gray', interpolation='nearest')
            fig.colorbar(im, ax=ax_hit)
            ax_hit.set_title(f"filter [{i},{j}]")
            ax_hit.set_xticks([])
            ax_hit.set_yticks([])

    fig.suptitle("Filters")
    fig.tight_layout()

    experiment.log_figure(figure_name=f'filters/conv',
                          figure=fig,
                          step=epoch)
    plt.clf()
    plt.close()

    return fig, plt

def log_embedding_histograms(epoch, loader, split='train'):
    """
    For each ORIGINAL class in `loader`, collect encoder embeddings,
    flatten to 1-D values, and plot a histogram.
    Upload directly to Comet without saving locally.
    """
    if not (args.use_comet and experiment):
        return  # no-op if Comet is off

    model.eval()
    device = torch.device("cuda" if args.cuda else "cpu")

    bucket = defaultdict(list)

    with torch.no_grad():
        for data, _, orig_labels, _ in loader:
            data = data.to(device)
            logits, emb = model(data, epoch, experiment)  # emb shape [B, D]
            emb_flat = emb.detach().cpu().view(emb.size(0), -1)  # [B, D]

            if isinstance(orig_labels, torch.Tensor):
                ol = orig_labels.detach().cpu().tolist()
            else:
                ol = list(map(int, orig_labels))

            for vec, lbl in zip(emb_flat, ol):
                bucket[int(lbl)].append(vec.numpy())

    # Make a histogram per class
    for lbl, chunks in bucket.items():
        if not chunks:
            continue
        vals = np.concatenate(chunks, axis=0)

        plt.figure()
        plt.hist(vals, bins=50)
        title = f"{label_to_name.get(lbl, str(lbl))} — embedding values (epoch {epoch})"
        plt.title(title)
        plt.xlabel("Embedding value")
        plt.ylabel("Count")

        # Save to buffer instead of disk
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        plt.close()

        # Upload to Comet
        experiment.log_image(
            buf,
            name=f"{split}_emb_hist_label{lbl}",
            step=epoch,
        )
        buf.close()

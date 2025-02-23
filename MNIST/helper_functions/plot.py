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
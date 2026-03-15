import numpy as np
import matplotlib.pyplot as plt

scatter_style = {
    "s": 60,
    #"color": 'royalblue',   #"blue", #navy
    "alpha": 0.85,
    "cmap": "viridis",
    "zorder": 2
}

bar_style = {
    "color": "skyblue",
    "edgecolor": "black",
    "linewidth": 1,
    "alpha": 0.9
}

line_style = {
    "linewidth": 2.5,
    "linestyle": "-",
    "color": "#1f77b4",
    #"marker": "o",
    "markersize": 6,
    "markerfacecolor": "white",
    "markeredgewidth": 1.5,
    "markeredgecolor": "#1f77b4",
    "alpha": 0.9
}

def apply_global_style():
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-colorblind")
    plt.grid(True, linestyle='--', zorder=-1, alpha=0.7)
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16
    })

def plot_comparison_w_heatmap(original, recreation, colormap='bwr', reconstruction_heatmap=False):
    
    residuals = np.abs(original - recreation)

    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    cmap = plt.get_cmap(colormap)
    colors = cmap(residuals)

    axes[0].bar(range(len(original)), original, zorder=1, color=colors if not reconstruction_heatmap else 'royalblue')
    axes[0].set_title(f"Original histogram")
    axes[0].set_xlabel("Bin")
    axes[0].set_ylabel("Value")

    axes[1].bar(range(len(recreation)), recreation, zorder=1, color=colors if reconstruction_heatmap else 'royalblue' )
    axes[1].set_title(f"Reconstructed histogram")
    axes[1].set_xlabel("Bin")
    axes[1].set_ylabel("Value")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt

    # Data points
    x = [1, 1.6, 1.4, 0.7]
    x1 = [3, 2.5, 2.1, 3.7]
    y_square = [1, 4.7, 4, 6]
    y_triangle = [2, 6, 4.8, 6.5]

    # Plotting squared points
    plt.scatter(x, y_square, color='blue', marker='s', label='Squares')

    # Plotting triangled points
    plt.scatter(x1, y_triangle, color='red', marker='^', label='Triangles')

    # Adding labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Squared and Triangled Points')
    plt.legend()

    # Show the plot
    plt.show()


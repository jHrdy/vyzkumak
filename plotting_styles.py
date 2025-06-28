scatter_style = {
    "s": 60,
    "color": 'royalblue',   #"blue", #navy
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
    "marker": "o",
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

if __name__ == '__main__':
    
    import numpy as np
    import matplotlib.pyplot as plt
    x, y = np.random.randn(2, 50), np.random.randn(2, 50)

    apply_global_style()
    #distances = np.array([1.2, 2.5, 0.9, 3.1])
    #x = np.arange(len(distances))
    #labels = [f'Bod {i}' for i in x]

    plt.scatter(x, y, **scatter_style)
    plt.show()

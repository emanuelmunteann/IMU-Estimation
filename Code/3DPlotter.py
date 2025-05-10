import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def read_csv_data(file_path):
    """Read CSV data from local storage"""
    return pd.read_csv(file_path)

def plot_3d_comparisons_mesh(data):
    metrics = ['RMSE', 'R^2']
    axes = ['X', 'Y', 'Z']
    colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'twilight']

    for idx, metric in enumerate(metrics):
        for jdx, axis in enumerate(axes):
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            col_name = f'{metric} Biceps_{axis}'

            x = data['NumberofTree']
            y = data['Random_State']
            z = data[col_name]

            x_unique = np.sort(x.unique())
            y_unique = np.sort(y.unique())
            X, Y = np.meshgrid(x_unique, y_unique)

            Z = np.empty_like(X, dtype=float)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    mask = (x == X[i, j]) & (y == Y[i, j])
                    if mask.any():
                        Z[i, j] = z[mask].values
                    else:
                        Z[i, j] = np.nan

            surf = ax.plot_surface(X, Y, Z, cmap=colormaps[idx*3 + jdx], edgecolor='k')
            ax.set_xlabel('Number of Trees', fontsize=10)
            ax.set_ylabel('Random State', fontsize=10)
            ax.set_zlabel(col_name, fontsize=10)
            ax.set_title(f'{col_name} vs Trees & Random State', fontsize=12)
            cbar = fig.colorbar(surf, shrink=0.6)
            cbar.set_label(col_name, rotation=270, labelpad=15)
            plt.show(block=False)
    plt.show()

# Usage example:
data = read_csv_data('Tree_si_RandomState.csv')
plot_3d_comparisons_mesh(data)

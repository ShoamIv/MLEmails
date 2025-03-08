import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

def pca(datastore_path, figure_folder, file, output):
    # Load DataFrame
    df = pd.read_csv(f"{datastore_path}/{file}")

    # Extract features and labels
    X = df.drop(columns=['label'])  # Features: all except 'label'
    y = df['label']  # Target variable

    # Standardize and reduce to 2D using PCA
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    X_reduced_2d = PCA(n_components=2).fit_transform(X_standardized)
    df_2d = pd.DataFrame(X_reduced_2d, columns=['PC1', 'PC2'])
    df_2d['Label'] = y

    plt.figure(figsize=(8, 6))
    for label in df_2d['Label'].unique():
        subset = df_2d[df_2d['Label'] == label]
        plt.scatter(subset['PC1'], subset['PC2'], label=str(label), alpha=0.7)

    plt.title(output+" 2D PCA Visualization of Email Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Label")
    plt.grid(True)
    plt.savefig(os.path.join(figure_folder, output+"2DPCA.png"), bbox_inches="tight")
    plt.close()

    # Reduce to 3D
    X_reduced_3d = PCA(n_components=3).fit_transform(X_standardized)

    # Convert to a DataFrame
    df_3d = pd.DataFrame(X_reduced_3d, columns=['PC1', 'PC2', 'PC3'])
    df_3d['Label'] = y

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each label category with different colors
    for label in df_3d['Label'].unique():
        subset = df_3d[df_3d['Label'] == label]
        ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'], label=str(label), alpha=0.7)

    ax.set_title(output+" 3D PCA Visualization of Email Data")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.legend(title="Label")

    plt.savefig(os.path.join(figure_folder, output+"3DPCA.png"), bbox_inches="tight")
    plt.close()

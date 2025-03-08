import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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

    plt.title("2D PCA Visualization of Email Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Label")
    plt.grid(True)
    plt.savefig(os.path.join(figure_folder, output), bbox_inches="tight")
    plt.close()

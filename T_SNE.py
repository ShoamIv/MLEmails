import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def tsne(datastore_path, figure_folder, file, output):
    # Load DataFrame
    df = pd.read_csv(f"{datastore_path}/{file}")

    # Extract features and labels
    X = df.drop(columns=['label'])  # Features: all except 'label'
    y = df['label']  # Target variable

    # Standardize and reduce to 2D using PCA
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # Apply PCA for dimensionality reduction (50 components)
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_standardized)

    # Apply t-SNE for further reduction to 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="random", learning_rate=200)
    X_tsne = tsne.fit_transform(X_pca)

    # Create DataFrame for t-SNE visualization
    df_tsne = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])
    df_tsne["Label"] = y

    # Create a scatter plot for t-SNE
    plt.figure(figsize=(8, 6))
    for label in df_tsne["Label"].unique():
        subset = df_tsne[df_tsne["Label"] == label]
        plt.scatter(subset["TSNE1"], subset["TSNE2"], label=str(label), alpha=0.7)

    plt.title("t-SNE Visualization of Email Data (Standardized + PCA Preprocessing)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Email Category")
    plt.grid(True)
    plt.savefig(os.path.join(figure_folder, output), bbox_inches="tight")
    plt.close()

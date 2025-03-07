import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import re


def load_filtered_emails(datastore_path):
    return pd.read_csv(os.path.join(datastore_path, "filtered_emails.csv"))


def train_folder_embeddings(filtered_emails, datastore_path):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Efficient BERT model
    folder_names = filtered_emails['X-Folder'].unique().tolist()
    folder_embeddings = model.encode(folder_names, convert_to_numpy=True)

    df = pd.DataFrame(folder_embeddings, index=folder_names)
    df.to_csv(os.path.join(datastore_path, "folder_embeddings_bert.csv"), index=True)
    return df, model


def plot_tsne(folder_embeddings, figure_folder):
    num_samples = len(folder_embeddings)
    if num_samples < 2:
        print("Not enough samples for t-SNE visualization.")
        return

    perplexity = min(5, num_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_results = tsne.fit_transform(folder_embeddings)
    tsne_df = pd.DataFrame(tsne_results, columns=["TSNE 1", "TSNE 2"], index=folder_embeddings.index)

    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_df["TSNE 1"], tsne_df["TSNE 2"], color='skyblue', alpha=0.7)
    for folder, (x, y) in tsne_df.iterrows():
        plt.text(x, y, folder, fontsize=8, ha='right')
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE Projection of Folder Embeddings (BERT)")
    plt.grid(True)
    plt.savefig(os.path.join(figure_folder, "TSNE_BERT.png"), bbox_inches="tight")


def Run_BertEmbedding(datastore_path, figure_folder):
    filtered_emails = load_filtered_emails(datastore_path)
    folder_embeddings, model = train_folder_embeddings(filtered_emails, datastore_path)
    plot_tsne(folder_embeddings, figure_folder)



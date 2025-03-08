import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from nltk.corpus import stopwords


def load_filtered_emails(datastore_path):
    return pd.read_csv(os.path.join(datastore_path, "filtered_emails.csv"))


# Preprocessing function
def preprocess_text(text, stop_words):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text


def train_folder_embeddings(filtered_emails, datastore_path, stop_words):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Efficient BERT model
    folder_names = filtered_emails['X-Folder'].apply(lambda x: preprocess_text(x, stop_words)).unique().tolist()
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


def make_emails_embeddings(filtered_emails, datastore_path, stop_words):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    filtered_emails['Subject_Embedding'] = filtered_emails['Subject'].apply(
        lambda x: preprocess_text(x, stop_words)).apply(model.encode)
    filtered_emails['Body_Embedding'] = filtered_emails['Body'].apply(lambda x: preprocess_text(x, stop_words)).apply(
        model.encode)

    flattened_subject_embeddings = pd.DataFrame(filtered_emails["Subject_Embedding"].tolist(),
                                                index=filtered_emails.index)
    flattened_body_embeddings = pd.DataFrame(filtered_emails["Body_Embedding"].tolist(), index=filtered_emails.index)

    flattened_subject_embeddings.columns = [f"subject_embedding_{i}" for i in range(384)]
    flattened_body_embeddings.columns = [f"body_embedding_{i}" for i in range(384)]

    final_df = pd.concat([filtered_emails, flattened_body_embeddings, flattened_subject_embeddings], axis=1)
    final_df.drop(
        columns=["message", "From", "To", "Subject", "Body", "Body_Embedding",
                 "Subject_Embedding"], inplace=True)

    folder_mapping = {"personal": 0, "hr": 1, "meetings and scheduling": 2, "operations and logistics": 3,
                      "projects": 4, "corporate and legal": 5, "finance": 6}

    final_df["label"] = final_df["X-Folder"].map(folder_mapping)
    final_df.dropna(subset=["label"], inplace=True)

    final_df["label"] = final_df["label"].astype(int)
    final_df.drop(columns=["X-Folder"], inplace=True)

    final_df.to_csv(os.path.join(datastore_path, "final_emails_bert_embeddings.csv"), index=False)


def Run_BertEmbedding(datastore_path, figure_folder):
    filtered_emails = load_filtered_emails(datastore_path)
    stop_words = set(stopwords.words("english"))
    folder_embeddings, model = train_folder_embeddings(filtered_emails, datastore_path, stop_words)
    plot_tsne(folder_embeddings, figure_folder)
    make_emails_embeddings(filtered_emails, datastore_path, stop_words)

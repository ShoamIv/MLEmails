import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import re
from nltk.corpus import stopwords


# Load filtered emails
def load_filtered_emails(datastore_path):
    return pd.read_csv(os.path.join(datastore_path, "filtered_emails.csv"))


# Train BERT embeddings for folder names
def train_folder_embeddings(filtered_emails, datastore_path):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Efficient BERT model
    folder_names = filtered_emails['X-Folder'].unique().tolist()
    folder_embeddings = model.encode(folder_names, convert_to_numpy=True)

    df = pd.DataFrame(folder_embeddings, index=folder_names)
    df.to_csv(os.path.join(datastore_path, "folder_embeddings_bert.csv"), index=True)
    return df, model


# Plot t-SNE visualization for folder embeddings
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


# Clean and tokenize text
def clean_and_tokenize_text(text, stop_words):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = word_tokenize(text.lower())
    return " ".join([word for word in words if word not in stop_words])


# Train BERT embeddings for email body and subject
def train_email_embeddings(filtered_emails, datastore_path, model):
    stop_words = set(stopwords.words("english"))
    filtered_emails["Tokenized_Body"] = filtered_emails["Body"].apply(lambda x: clean_and_tokenize_text(x, stop_words))
    filtered_emails["Tokenized_Subject"] = filtered_emails["Subject"].apply(
        lambda x: clean_and_tokenize_text(x, stop_words))

    # Encode email bodies and subjects using BERT
    body_embeddings = model.encode(filtered_emails["Tokenized_Body"].tolist(), convert_to_numpy=True)
    subject_embeddings = model.encode(filtered_emails["Tokenized_Subject"].tolist(), convert_to_numpy=True)

    # Save embeddings
    np.save(os.path.join(datastore_path, "body_embeddings_bert.npy"), body_embeddings)
    np.save(os.path.join(datastore_path, "subject_embeddings_bert.npy"), subject_embeddings)

    return body_embeddings, subject_embeddings


# Process and save final email data with BERT embeddings
def process_and_save_final_data(filtered_emails, body_embeddings, subject_embeddings, datastore_path):
    # Add embeddings to the DataFrame
    body_embedding_df = pd.DataFrame(body_embeddings, columns=[f"body_embedding_{i}" for i in range(body_embeddings.shape[1])])
    subject_embedding_df = pd.DataFrame(subject_embeddings, columns=[f"subject_embedding_{i}" for i in range(subject_embeddings.shape[1])])

    final_df = pd.concat([filtered_emails, body_embedding_df, subject_embedding_df], axis=1)

    # Drop unnecessary columns
    final_df.drop(
        columns=["message", "From", "To", "Subject", "Body", "Tokenized_Body", "Tokenized_Subject"], inplace=True)

    # Map folder names to labels
    folder_mapping = {"personal": 0, "hr": 1, "meetings and scheduling": 2, "operations and logistics": 3,
                      "projects": 4, "corporate and legal": 5, "finance": 6}
    final_df["label"] = final_df["X-Folder"].map(folder_mapping)
    final_df.dropna(subset=["label"], inplace=True)
    final_df["label"] = final_df["label"].astype(int)
    final_df.drop(columns=["X-Folder"], inplace=True)

    # Save final data
    final_df.to_csv(os.path.join(datastore_path, "final_emails_bert.csv"), index=False)


# Main function to execute BERT embedding stage
def Run_BertEmbedding(datastore_path, figure_folder):
    filtered_emails = load_filtered_emails(datastore_path)
    print("Unique folders found:", filtered_emails['X-Folder'].unique())

    # Train folder embeddings
    folder_embeddings, model = train_folder_embeddings(filtered_emails, datastore_path)
    plot_tsne(folder_embeddings, figure_folder)

    # Train email body and subject embeddings
    body_embeddings, subject_embeddings = train_email_embeddings(filtered_emails, datastore_path, model)

    # Process and save final data
    process_and_save_final_data(filtered_emails, body_embeddings, subject_embeddings, datastore_path)
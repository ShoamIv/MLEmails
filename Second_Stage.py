import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk import word_tokenize
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
import re


# Load filtered emails
def load_filtered_emails(datastore_path):
    return pd.read_csv(os.path.join(datastore_path, "filtered_emails.csv"))


# Train Word2Vec model on folder names
def train_folder_embeddings(filtered_emails, datastore_path):
    tokenized_folders = [word_tokenize(folder.lower()) for folder in filtered_emails['X-Folder'].unique()]
    word2vec_model = Word2Vec(sentences=tokenized_folders, vector_size=300, window=5, min_count=1, workers=4)
    folder_embeddings = make_vector_rep_df(filtered_emails, word2vec_model)
    folder_embeddings.to_csv(os.path.join(datastore_path, "folder_embeddings.csv"), index=False)
    return folder_embeddings, word2vec_model


# Convert words to vector representations
def vector_rep(word, word2vec_model):
    if word in word2vec_model.wv.key_to_index:
        return word2vec_model.wv[word]
    return np.zeros(shape=(300,))


# Compute average vector for a phrase
def general_vector_rep(phrase, word2vec_model):
    tokenized = word_tokenize(phrase)
    if not tokenized:
        return np.zeros(shape=(300,))
    vectors = np.array([vector_rep(word, word2vec_model) for word in tokenized])
    return np.mean(vectors, axis=0)


# Create a DataFrame for vector representations
def make_vector_rep_df(filtered_emails, word2vec_model):
    labels = list(filtered_emails['X-Folder'].unique())
    vectors_dict = {label: general_vector_rep(label, word2vec_model) for label in labels}
    df = pd.DataFrame.from_dict(vectors_dict, orient='index')
    df.fillna(0, inplace=True)
    return df


# Plot t-SNE visualization
def plot_tsne(folder_embeddings, figure_folder):
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_results = tsne.fit_transform(folder_embeddings)
    tsne_df = pd.DataFrame(tsne_results, columns=["TSNE 1", "TSNE 2"], index=folder_embeddings.index)

    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_df["TSNE 1"], tsne_df["TSNE 2"], color='skyblue', alpha=0.7)
    for folder, (x, y) in tsne_df.iterrows():
        plt.text(x, y, folder, fontsize=8, ha='right')
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE Projection of Folder Embeddings")
    plt.grid(True)
    plt.savefig(os.path.join(figure_folder, "TSNE"), bbox_inches="tight")


# Clean and tokenize text
def clean_and_tokenize_text(text, stop_words):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = word_tokenize(text.lower())
    return " ".join([word for word in words if word not in stop_words])


# Train Word2Vec models for email body and subject
def train_email_embeddings(filtered_emails, datastore_path):
    stop_words = set(stopwords.words("english"))
    filtered_emails["Tokenized_Body"] = filtered_emails["Body"].apply(lambda x: clean_and_tokenize_text(x, stop_words))
    filtered_emails["Tokenized_Subject"] = filtered_emails["Subject"].apply(
        lambda x: clean_and_tokenize_text(x, stop_words))

    word2vec_model_body = Word2Vec(sentences=filtered_emails["Tokenized_Body"], vector_size=300, window=5, min_count=2,
                                   workers=4, sg=1)
    word2vec_model_subject_50 = Word2Vec(sentences=filtered_emails["Tokenized_Subject"], vector_size=50, window=5,
                                         min_count=2, workers=4, sg=1)

    word2vec_model_body.save(os.path.join(datastore_path, "word2vec_email_body.model"))
    word2vec_model_subject_50.save(os.path.join(datastore_path, "word2vec_email_subject_50.model"))

    return word2vec_model_body, word2vec_model_subject_50


# Compute document embeddings
def document_embedding(tokenized_text, model, size):
    vectors = [model.wv[word] for word in tokenized_text if word in model.wv]
    return sum(vectors) / len(vectors) if vectors else [0] * size


# Process and save final email data
def process_and_save_final_data(filtered_emails, word2vec_model_body, word2vec_model_subject_50, datastore_path):
    filtered_emails["Body_Embedding"] = filtered_emails["Tokenized_Body"].apply(
        lambda x: document_embedding(x, word2vec_model_body, 300))
    filtered_emails["Subject_Embedding"] = filtered_emails["Tokenized_Subject"].apply(
        lambda x: document_embedding(x, word2vec_model_subject_50, 50))

    flattened_body_embeddings = pd.DataFrame(filtered_emails["Body_Embedding"].tolist(), index=filtered_emails.index)
    flattened_subject_embeddings = pd.DataFrame(filtered_emails["Subject_Embedding"].tolist(),
                                                index=filtered_emails.index)

    flattened_body_embeddings.columns = [f"body_embedding_{i}" for i in range(300)]
    flattened_subject_embeddings.columns = [f"subject_embedding_{i}" for i in range(50)]

    final_df = pd.concat([filtered_emails, flattened_body_embeddings, flattened_subject_embeddings], axis=1)
    final_df.drop(
        columns=["message", "From", "To", "Subject", "Body", "Tokenized_Body", "Tokenized_Subject", "Body_Embedding",
                 "Subject_Embedding"], inplace=True)

    folder_mapping = {"personal": 0, "hr": 1, "meetings and scheduling": 2, "operations and logistics": 3,
                      "projects": 4, "corporate and legal": 5, "finance": 6}

    final_df["label"] = final_df["X-Folder"].map(folder_mapping)
    final_df.dropna(subset=["label"], inplace=True)

    final_df["label"] = final_df["label"].astype(int)
    final_df.drop(columns=["X-Folder"], inplace=True)

    final_df.to_csv(os.path.join(datastore_path, "final_emails_50.csv"), index=False)


# Main function to execute second stage
def Run_SecondStage(datastore_path, figure_folder):
    filtered_emails = load_filtered_emails(datastore_path)
    print("Unique folders found:", filtered_emails['X-Folder'].unique())

    folder_embeddings, word2vec_model = train_folder_embeddings(filtered_emails, datastore_path)
    plot_tsne(folder_embeddings, figure_folder)
    word2vec_model_body, word2vec_model_subject_50 = train_email_embeddings(filtered_emails, datastore_path)
    process_and_save_final_data(filtered_emails, word2vec_model_body, word2vec_model_subject_50, datastore_path)

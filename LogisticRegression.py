import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Function to train Logistic Regression model
def train_logistic_regression(datastore_path, file, figure_folder):
    # Load your DataFrame here
    df = pd.read_csv(f"{datastore_path}/{file}")

    # Extract features and labels
    X = df.drop(columns=['label'])  # Features: all except 'label'
    y = df['label']  # Target variable

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert integer labels to one-hot encoded labels
    y_train = keras.utils.to_categorical(y_train, num_classes=7)
    y_test = keras.utils.to_categorical(y_test, num_classes=7)

    # Define the logistic regression model
    model = keras.Sequential()
    model.add(
        keras.layers.Dense(7, activation='softmax', input_shape=(X_train.shape[1],)))  # Corrected input_shape parameter

    # Create an Adam optimizer with a custom learning rate
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Define batch size and number of epochs
    batch_size = 100
    epochs = 112

    # Train the model with mini-batches
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

    # Evaluate the model on the test set
    y_test_pred = model.predict(X_test)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)  # Convert probabilities to class labels
    y_test_true_classes = np.argmax(y_test, axis=1)  # Convert one-hot encoded labels back to integers
    test_accuracy = accuracy_score(y_test_true_classes, y_test_pred_classes)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot the training and test data losses
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(history.history['loss'], 'b', label='Train')
    ax.plot(history.history['val_loss'], 'r', label='Test')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(os.path.join(figure_folder, "LG_loss.png"), bbox_inches="tight")
    plt.close()

    # Standardize and reduce to 2D using PCA
    X_standardized = scaler.fit_transform(X)
    X_reduced = PCA(n_components=2).fit_transform(X_standardized)

    # Convert to a DataFrame for visualization
    df_pca = pd.DataFrame(X_reduced, columns=['PC1', 'PC2'])
    df_pca['Label'] = y  # Add labels

    # Create a scatter plot for PCA
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', hue=df_pca['Label'].astype(str), palette='viridis', data=df_pca, alpha=0.7)
    plt.title("PCA Visualization of Email Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Label")
    plt.grid(True)
    plt.savefig(os.path.join(figure_folder, "LG_PCA_sns.png"), bbox_inches="tight")
    plt.close()

    # Reduce to 2D for static visualization
    X_reduced_2d = PCA(n_components=2).fit_transform(X_standardized)
    df_2d = pd.DataFrame(X_reduced_2d, columns=['PC1', 'PC2'])
    df_2d['Label'] = y

    # Create a simple scatter plot
    plt.figure(figsize=(8, 6))
    for label in df_2d['Label'].unique():
        subset = df_2d[df_2d['Label'] == label]
        plt.scatter(subset['PC1'], subset['PC2'], label=str(label), alpha=0.7)

    plt.title("2D PCA Visualization of Email Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Label")
    plt.grid(True)
    plt.savefig(os.path.join(figure_folder, "LG_PCA.png"), bbox_inches="tight")
    plt.close()

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
    plt.savefig(os.path.join(figure_folder, "LG_t-SNE.png"), bbox_inches="tight")
    plt.close()

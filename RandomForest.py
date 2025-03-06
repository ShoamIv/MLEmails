import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def train_random_forest(datastore_path, figure_folder):
    # Load the dataset
    df = pd.read_csv(f"{datastore_path}/final_emails_50.csv")

    # Extract features and labels
    X = df.drop(columns=['label'])  # Features: all except 'label'
    y = df['label']  # Target variable

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest Classifier
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')

    # Train the model
    rf_model.fit(X_train, y_train)
    print("Random Forest model training completed")

    # Predict on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Random Forest Accuracy: {accuracy:.4f}')

    # Confusion Matrix
    class_names = sorted(df['label'].unique())
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title('Confusion Matrix for Multi-Class Random Forest')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, "RF_confusion_matrix.png"), bbox_inches="tight")

    # Classification Report
    print("Classification Report:")
    report = classification_report(y_test, y_pred)
    print(report)

    # Save classification report to a text file
    with open(os.path.join(figure_folder, "RF_classification_report.txt"), "w") as f:
        f.write(report)

    # Plot feature importance
    feature_importance = rf_model.feature_importances_
    feature_names = X.columns

    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)

    # Plot top 20 important features
    plt.figure(figsize=(10, 8))
    plt.barh(range(20), feature_importance[sorted_idx[-20:]])
    plt.yticks(range(20), [feature_names[i] for i in sorted_idx[-20:]])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Important Features in Random Forest')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, "RF_feature_importance.png"), bbox_inches="tight")

    # Plot accuracy
    plt.figure(figsize=(6, 4))
    plt.bar(['Random Forest'], [accuracy])
    plt.ylim(0, 1)
    plt.title('Random Forest Model Accuracy')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(figure_folder, "RF_accuracy.png"), bbox_inches="tight")

    # Standardize data for visualizations
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # PCA visualization
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
    plt.savefig(os.path.join(figure_folder, "RF_PCA_sns.png"), bbox_inches="tight")

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
    plt.savefig(os.path.join(figure_folder, "RF_t-SNE.png"), bbox_inches="tight")

    return rf_model
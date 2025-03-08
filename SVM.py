import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


def train_SVM(datastore_path, figure_folder):
    # Load your DataFrame here
    df = pd.read_csv(f"{datastore_path}/final_emails_50.csv")

    # Extract features and labels
    X = df.drop(columns=['label'])  # Features: all except 'label'
    y = df['label']  # Target variable

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and train the Linear SVM model
    print("Training Linear SVM model...")
    svm = LinearSVC(max_iter=10000, random_state=42, dual=False, tol=0.0001)
    ova_classifier = OneVsRestClassifier(svm)
    ova_classifier.fit(X_train, y_train)
    print("Linear SVM model training completed")

    # Make predictions
    print("Making predictions on test set...")
    y_pred = ova_classifier.predict(X_test)
    print(f"Predictions shape: {y_pred.shape}")

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Linear SVM Accuracy: {accuracy:.4f}')

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Linear SVM Confusion Matrix')
    plt.savefig(os.path.join(figure_folder, "SVM_confusion_matrix.png"), bbox_inches="tight")
    plt.close()

    # Print the classification report
    print("Classification Report:")
    report = classification_report(y_test, y_pred)
    print(report)

    # Save classification report to a text file
    with open(os.path.join(figure_folder, "SVM_classification_report.txt"), "w") as f:
        f.write(report)

    # Plot the accuracy
    plt.figure(figsize=(6, 4))
    plt.bar(['Linear SVM'], [accuracy])
    plt.ylim(0, 1)
    plt.title('Linear SVM Model Accuracy')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(figure_folder, "SVM_accuracy.png"), bbox_inches="tight")
    plt.close()

    # Train RBF Kernel SVM
    print("Training RBF Kernel SVM model...")
    svm_rbf = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
    svm_rbf.fit(X_train, y_train)
    print("RBF Kernel SVM model training completed")

    # Make predictions with RBF kernel
    y_pred_rbf = svm_rbf.predict(X_test)
    accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
    print(f'RBF Kernel SVM Accuracy: {accuracy_rbf:.4f}')

    # Create confusion matrix for RBF
    cm_rbf = confusion_matrix(y_test, y_pred_rbf)

    # Plot the confusion matrix for RBF
    plt.figure(figsize=(8, 6))
    disp_rbf = ConfusionMatrixDisplay(confusion_matrix=cm_rbf)
    disp_rbf.plot(cmap=plt.cm.Blues)
    plt.title('RBF Kernel SVM Confusion Matrix')
    plt.savefig(os.path.join(figure_folder, "SVM_RBF_confusion_matrix.png"), bbox_inches="tight")
    plt.close()

    # Print the classification report for RBF
    print("RBF Classification Report:")
    report_rbf = classification_report(y_test, y_pred_rbf)
    print(report_rbf)

    # Save RBF classification report to a text file
    with open(os.path.join(figure_folder, "SVM_RBF_classification_report.txt"), "w") as f:
        f.write(report_rbf)

    # Compare both models
    plt.figure(figsize=(6, 4))
    plt.bar(['Linear SVM', 'RBF Kernel SVM'], [accuracy, accuracy_rbf])
    plt.ylim(0, 1)
    plt.title('SVM Models Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(figure_folder, "SVM_comparison.png"), bbox_inches="tight")
    plt.close()

    # Similar to the logistic regression: PCA visualization
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
    plt.savefig(os.path.join(figure_folder, "SVM_PCA_sns.png"), bbox_inches="tight")
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
    plt.savefig(os.path.join(figure_folder, "SVM_t-SNE.png"), bbox_inches="tight")
    plt.close()

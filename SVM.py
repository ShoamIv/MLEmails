import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


def train_SVM(datastore_path, figure_folder, file, output, output_report, embedding):

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

    # Create and train the Linear SVM model
    print("Training Linear SVM model...")
    svm = LinearSVC(max_iter=10000, dual=False, tol=0.0001)
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

    # Confusion Matrix
    folder_mapping = {"personal": 0, "hr": 1, "meetings and scheduling": 2, "operations and logistics": 3,
                      "projects": 4, "corporate and legal": 5, "finance": 6}

    class_names = [name for name, idx in sorted(folder_mapping.items(), key=lambda x: x[1])]
    cm = confusion_matrix(y_test, y_pred)
    # Plot the confusion matrix
    plt.figure(figsize=(16, 14))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(f'Confusion Matrix: {embedding} Linear SVM')
    plt.savefig(os.path.join(figure_folder, output), bbox_inches="tight")
    plt.close()
    # Print the classification report
    print("Classification Report:")
    report = classification_report(y_test, y_pred)
    print(report)

    # Save classification report to a text file
    with open(os.path.join(figure_folder, output_report), "w") as f:
        f.write(report)

    # Train RBF Kernel SVM
    output = "RBF"+output
    print("Training RBF Kernel SVM model...")
    svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')
    svm_rbf.fit(X_train, y_train)
    print("RBF Kernel SVM model training completed")

    # Make predictions with RBF kernel
    y_pred_rbf = svm_rbf.predict(X_test)
    accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
    print(f'RBF Kernel SVM Accuracy: {accuracy_rbf:.4f}')

    # Create confusion matrix for RBF
    cm_rbf = confusion_matrix(y_test, y_pred_rbf)

    # Plot the confusion matrix for RBF
    plt.figure(figsize=(16, 14))
    disp_rbf = ConfusionMatrixDisplay(confusion_matrix=cm_rbf, display_labels=class_names)
    disp_rbf.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(f'Confusion Matrix: {embedding} RBF Kernel SVM')
    plt.savefig(os.path.join(figure_folder, output), bbox_inches="tight")
    plt.close()

    # Print the classification report for RBF
    print("RBF Classification Report:")
    report_rbf = classification_report(y_test, y_pred_rbf)
    print(report_rbf)

    # Save RBF classification report to a text file
    output_report = "RBF" + output_report
    with open(os.path.join(figure_folder, output_report), "w") as f:
        f.write(report_rbf)



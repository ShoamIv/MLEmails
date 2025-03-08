import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_random_forest(datastore_path, figure_folder, file, output, output_report, embedding):

    # Load the dataset
    df = pd.read_csv(f"{datastore_path}/{file}")

    # Extract features and labels
    X = df.drop(columns=['label'])  # Features: all except 'label'
    y = df['label']  # Target variable

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the Random Forest Classifier
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=300, class_weight='balanced')

    # Train the model
    rf_model.fit(X_train, y_train)
    print("Random Forest model training completed")

    # Predict on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Random Forest Accuracy: {accuracy:.4f}')

    # Confusion Matrix
    folder_mapping = {"personal": 0, "hr": 1, "meetings and scheduling": 2, "operations and logistics": 3,
                      "projects": 4, "corporate and legal": 5, "finance": 6}

    class_names = [name for name, idx in sorted(folder_mapping.items(), key=lambda x: x[1])]
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(16, 14))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(f'Confusion Matrix: {embedding} Random Forest')
    plt.savefig(os.path.join(figure_folder, output), bbox_inches="tight")
    plt.close()

    # Classification Report
    print("Classification Report:")
    report = classification_report(y_test, y_pred)
    print(report)

    # Save classification report to a text file
    with open(os.path.join(figure_folder, output_report), "w") as f:
        f.write(report)

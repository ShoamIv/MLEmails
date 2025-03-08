import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split


def train_random_forest(datastore_path, figure_folder, file):

    # Load the dataset
    df = pd.read_csv(f"{datastore_path}/{file}")

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
    plt.close()

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
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(6, 4))
    plt.bar(['Random Forest'], [accuracy])
    plt.ylim(0, 1)
    plt.title('Random Forest Model Accuracy')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(figure_folder, "RF_accuracy.png"), bbox_inches="tight")
    plt.close()

    return rf_model

import os
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Function to train Logistic Regression model
def train_logistic_regression(datastore_path, figure_folder, file, output, output_report, embedding):
    # Load DataFrame here
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
    model.add(keras.layers.Dense(7, activation='softmax', input_shape=(X_train.shape[1],)))

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

    # Calculate accuracy
    test_accuracy = accuracy_score(y_test_true_classes, y_test_pred_classes)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot the training and test data losses
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(history.history['loss'], 'b', label='Train')
    ax.plot(history.history['val_loss'], 'r', label='Test')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title(f'{embedding} Logistic Regression Train Loss')
    ax.legend()
    plt.savefig(os.path.join(figure_folder, output), bbox_inches="tight")
    plt.close()

    # Generate and print classification report
    print("Classification Report:")
    report = classification_report(y_test_true_classes, y_test_pred_classes)
    print(report)

    # Save classification report to a text file
    with open(os.path.join(figure_folder, output_report), "w") as f:
        f.write(report)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from First_Stage import Run_FirstStage
from LogisticRegression import train_logistic_regression
from RandomForest import train_random_forest
from SVM import train_SVM
from Second_Stage import Run_SecondStage


def main():
    datastore_path = "."
    figure_folder = "Figure"

    print("Run First Stage")
    Run_FirstStage(datastore_path,figure_folder)

    print("Run Second Stage")
    Run_SecondStage(datastore_path, figure_folder)

    # Algorithms run on Word2vec
    print("\nTraining Logistic Regression...")
    train_logistic_regression(datastore_path, figure_folder)

    print("\nTraining Random Forest...")
    train_random_forest(datastore_path, figure_folder)

    print("\nTraining SVM...")
    train_SVM(datastore_path, figure_folder)

    # Bert model
    print("\nTraining On Bert Model...")

    # Algorithms
    # .....

if __name__ == '__main__':
    main()

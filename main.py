from BertEmbedding import Run_BertEmbedding
from LogisticRegression import train_logistic_regression


def main():
    datastore_path = "."
    figure_folder = "Figure"

    # print("Run First Stage")
    # Run_FirstStage(datastore_path, figure_folder)

    # print("Run Second Stage")
    # Run_SecondStage(datastore_path, figure_folder)
    #
    # # Algorithms run on Word2vec
    # print("\nTraining Logistic Regression...")
    # train_logistic_regression(datastore_path, 'final_emails_50.csv', figure_folder)
    #
    # print("\nTraining Random Forest...")
    # train_random_forest(datastore_path, figure_folder)
    #
    # print("\nTraining SVM...")
    # train_SVM(datastore_path, figure_folder)

    # Bert model
    print("\nTraining On Bert Model...")
    Run_BertEmbedding(datastore_path, figure_folder)

    # Algorithms run on Bert
    print("\nTraining Logistic Regression...")
    train_logistic_regression(datastore_path, 'final_emails_bert_embeddings.csv', figure_folder)


if __name__ == '__main__':
    main()

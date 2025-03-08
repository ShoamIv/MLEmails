from BertEmbedding import Run_BertEmbedding
from DataProcessing import preprocess_data
from LogisticRegression import train_logistic_regression
from RandomForest import train_random_forest
from SVM import train_SVM
from Create_Word2Vec import initiate_word2vec


def main():
    datastore_path = "."
    figure_folder = "Figure"

    print("Start processing the data...")
    # preprocess_data(datastore_path, figure_folder, )

    print("Initiating word2vec Embedding...")
    # initiate_word2vec(datastore_path, figure_folder)

    file = "final_emails_50.csv"

    # Algorithms run on Word2Vec

    print("\nRunning Logistic Regression on Word2Vec embeddings...")
    train_logistic_regression(datastore_path, figure_folder, file)

    print("\nRunning Random Forest on Word2Vec embeddings...")
    train_random_forest(datastore_path, figure_folder, file)

    print("\nRunning SVM on Word2Vec embeddings...")
    train_SVM(datastore_path, figure_folder, file)

    # Update file to Bert.
    file = "final_emails_bert_embeddings.csv"

    # Bert model
    print("\nTraining On Bert Model...")
    Run_BertEmbedding(datastore_path, figure_folder)

    # Algorithms run on Bert

    print("\nRunning Logistic Regression on BERT embeddings...")
    train_logistic_regression(datastore_path, figure_folder, file)

    print("\nRunning Random Forest on BERT embeddings...")
    train_random_forest(datastore_path, figure_folder, file)

    print("\nRunning SVM on BERT embeddings...")
    train_SVM(datastore_path, figure_folder, file)


if __name__ == '__main__':
    main()

from BertEmbedding import Run_BertEmbedding
from Create_Word2Vec import initiate_word2vec
from DataProcessing import preprocess_data
from LogisticRegression import train_logistic_regression
from PCA import pca
from RandomForest import train_random_forest
from SVM import train_SVM
from T_SNE import tsne


def main():
    datastore_path = "."
    figure_folder = "Figure"

    print("Start processing the data...")
    preprocess_data(datastore_path, figure_folder, )

    print("Initiating Word2Vec Embedding...")
    initiate_word2vec(datastore_path, figure_folder)

    file = "final_emails_50.csv"

    print("\nInitiating Word2Vec PCA and T-SNE...")
    pca(datastore_path, figure_folder, file, "Word2Vec")
    tsne(datastore_path, figure_folder, file, "T-SNE_Word2Vec.png")

    # Algorithms run on Word2Vec

    print("\nRunning Logistic Regression on Word2Vec embeddings...")
    train_logistic_regression(datastore_path, figure_folder, file, "W2V_LG_loss.png")

    print("\nRunning Random Forest on Word2Vec embeddings...")
    train_random_forest(datastore_path, figure_folder, file, "W2V_RF_confusion_matrix.png",
                        "W2VRF_classification_report.txt")

    print("\nRunning SVM on Word2Vec embeddings...")
    train_SVM(datastore_path, figure_folder, file, "W2V_SVM_confusion_matrix.png", "W2VSVM_classification_report.txt", )

    # Update file to Bert.
    file = "final_emails_bert_embeddings.csv"

    print("\nInitiating BERT PCA and T-SNE...")
    pca(datastore_path, figure_folder, file, "BERT")
    tsne(datastore_path, figure_folder, file, "T-SNE_BERT.png")

    # Bert model
    print("\nTraining On Bert Model...")
    Run_BertEmbedding(datastore_path, figure_folder)

    # Algorithms run on Bert

    print("\nRunning Logistic Regression on BERT embeddings...")
    train_logistic_regression(datastore_path, figure_folder, file, "BERT_LG_loss.png")

    print("\nRunning Random Forest on BERT embeddings...")
    train_random_forest(datastore_path, figure_folder, file, "BERT_RF_confusion_matrix.png",
                        "BERTRF_classification_report.txt")

    print("\nRunning SVM on BERT embeddings...")
    train_SVM(datastore_path, figure_folder, file, "BERT_SVM_confusion_matrix.png", "BERTSVM_classification_report.txt")


if __name__ == '__main__':
    main()

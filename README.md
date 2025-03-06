# Email Classification Project

## Introduction
This project aims to provide a gateway to a more effective solution for email categorization. Today, most email classification systems focus on broad categories like spam, junk, or promotional emails. However, our goal is to enable a more meaningful categorization—distinguishing between work-related emails, personal messages, and more.

In particular, within the work category, we aim to further refine classification by organizing emails into specific, meaningful subcategories. Using the Enron dataset as our foundation, we focus on developing a structured approach to classify work-related emails into distinct and relevant categories, enhancing both accessibility and efficiency in email management.

## Dataset & Preprocessing

The dataset initially contained approximately 500,000 emails, stored in various folders that served as initial labels. However, we identified that some folders lacked clear semantic focus, leading to inconsistent labeling and dataset imbalance. To improve data quality:

We removed generic folders such as 'Inbox,' 'Calendar,' and 'Junk File,' which contained disproportionately large numbers of emails but lacked meaningful distinctions.

We removed folders with very few emails, as they were unsuitable for training a robust classification model.

The dataset was significantly reduced, dropping from 500,000 emails to 64,512. To address this, we targeted folders with a significant number of emails and related names, combining them into 9 categories that we deemed fitting as labels for the emails. These categories were chosen based on their semantic relevance and the volume of emails they contained, ensuring a more balanced and meaningful dataset for our exploration of email categorization.

After examination, we dropped 2 categories: It and  Archive and Miscellaneous folders, as they lacked semantic relevance, leaving us with the following folders as categories:
personal, hr, meetings and scheduling, operations and logistics, projects,corporate and legal, finance.

The final dataset for model training consisted of 16,769 emails.

# Emails were preprocessed by:

Cleaning subject lines.

Standardizing text formats.

Removing stopwords and unnecessary metadata.

Dimensionality Reduction

To improve model efficiency and visualization, we applied dimensionality reduction techniques:

Principal Component Analysis (PCA) to reduce features to 2D and 3D representations.

t-Distributed Stochastic Neighbor Embedding (t-SNE) for visualizing high-dimensional folder embeddings.

# Features:
1.Content Length – The total length of the email body.

2.Number of Recipients – The count of recipients in the email.

3.Email Subject Embedding – A numerical representation of the email subject.

4.Email Body Embedding – A numerical representation of the email body.

For the last two features, we experimented with different embedding techniques. Initially, we used Word2Vec, but the results were not satisfactory. To improve performance, we turned to BERT (Bidirectional Encoder Representations from Transformers). The difference was significant, as demonstrated in the figures.

## Machine Learning Models Used

To classify emails effectively, we experimented with multiple models:

Random Forest: Used for classification tasks due to its robustness against overfitting and high interpretability.

Regression Model: Applied to predict certain numerical properties of emails based on extracted features.

Singular Value Decomposition (SVD) + ML Classifier: Utilized for reducing text data dimensionality before classification.

## Visualizations

To analyze and interpret the dataset, we used:

Matplotlib for static visualizations.

Seaborn for enhanced scatter plots.

Bar charts to visualize folder distributions before and after data cleaning.

Key Findings & Impact

Removing noisy folders improved classification accuracy by reducing ambiguity.

Dimensionality reduction techniques helped in better feature extraction and model performance.

SVM RBF Kernel Model performed best among the classifiers, balancing accuracy and efficiency.

We observed that Word2Vec did not provide satisfactory separation or model results for our email categorization task. Upon further research, we discovered that Word2Vec generates word vectors in a way that does not always preserve the semantic meaning of words, especially in contexts where word order and deeper contextual understanding are crucial.

To address this, we transitioned to using BERT (Bidirectional Encoder Representations from Transformers) for vectorizing the email text. Unlike Word2Vec, BERT captures the contextual meaning of words by considering the entire sentence structure, resulting in richer and more semantically meaningful embeddings.

Additionally, BERT generates higher-dimensional vectors, which provide a more nuanced representation of the text. This shift allowed us to better capture the semantic relationships between words and phrases in the emails, ultimately improving the performance of our categorization models.

## Future Improvements

Fine-tuning hyperparameters for better model optimization.

Expanding the dataset with external corpora to enhance model generalizability.

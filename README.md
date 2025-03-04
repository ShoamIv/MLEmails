# Email Classification ML Project

## Overview

This project focuses on classifying emails based on their content using Machine Learning (ML) techniques. The dataset is derived from the Enron email corpus, and the goal is to assign meaningful categories to emails by analyzing their textual features.

## Dataset & Preprocessing

"The dataset initially contained approximately 500,000 emails, stored in various folders that served as initial labels. However, we identified that some folders lacked clear semantic focus, leading to inconsistent labeling and dataset imbalance. To improve data quality:

We removed generic folders such as 'Inbox,' 'Calendar,' and 'Junk File,' which contained disproportionately large numbers of emails but lacked meaningful distinctions.

We removed folders with very few emails, as they were unsuitable for training a robust classification model.

The dataset was significantly reduced, dropping from 500,000 emails to 64,512. To address this, we targeted folders with a significant number of emails and related names, combining them into 9 categories that we deemed fitting as labels for the emails. These categories were chosen based on their semantic relevance and the volume of emails they contained, ensuring a more balanced and meaningful dataset for our exploration of email categorization.

After examination, we dropped 2 categories as they lacked semantic relevance, leaving us with 7 categories. The final dataset for model training consisted of 16,769 emails.

## Emails were preprocessed by:

Cleaning subject lines.

Standardizing text formats.

Removing stopwords and unnecessary metadata.

Dimensionality Reduction

To improve model efficiency and visualization, we applied dimensionality reduction techniques:

Principal Component Analysis (PCA) to reduce features to 2D and 3D representations.

Singular Value Decomposition (SVD) for improved text feature extraction.

t-Distributed Stochastic Neighbor Embedding (t-SNE) for visualizing high-dimensional folder embeddings.

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

The Random Forest model & SVM RBF Kernel Model performed best among the classifiers, balancing accuracy and efficiency.

We observed that Word2Vec did not provide satisfactory separation or model results for our email categorization task. Upon further research, we discovered that Word2Vec generates word vectors in a way that does not always preserve the semantic meaning of words, especially in contexts where word order and deeper contextual understanding are crucial.

To address this, we transitioned to using BERT (Bidirectional Encoder Representations from Transformers) for vectorizing the email text. Unlike Word2Vec, BERT captures the contextual meaning of words by considering the entire sentence structure, resulting in richer and more semantically meaningful embeddings.

Additionally, BERT generates higher-dimensional vectors, which provide a more nuanced representation of the text. This shift allowed us to better capture the semantic relationships between words and phrases in the emails, ultimately improving the performance of our categorization models.

## Future Improvements

Fine-tuning hyperparameters for better model optimization.

Expanding the dataset with external corpora to enhance model generalizability.

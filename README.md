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

The final dataset for model training consisted of 14,081 emails.

### Emails were preprocessed by:

Cleaning subject lines.

Standardizing text formats.

Removing stopwords and unnecessary metadata.

Dimensionality Reduction

To improve model efficiency and visualization, we applied dimensionality reduction techniques:

Principal Component Analysis (PCA) to reduce features to 2D and 3D representations.

t-Distributed Stochastic Neighbor Embedding (t-SNE) for visualizing high-dimensional folder embeddings.

## Labels
Owing to the fact that the emails in Enron dataset(emails.csv) are already organized into their respective folders, we leveraged word embeddings on the folder names to enhance our analysis. This approach was particularly effective because the folder names inherently capture the context and categorization of the emails. To streamline the process, we performed significant manual work to consolidate similar folders. For instance, folders like HR, HR_r, and HR_recruiting were combined into a single category, HR.
This approach not only preserved the organizational structure of the emails but also allowed us to leverage the semantic meaning embedded in the folder names, providing a robust foundation for our machine learning pipeline.
The final categories, which now serve as our labels, are as follows:

1.Personal

2.Operations and Logistics

3.Corporate and Legal

4.Finance

5.HR

6.Meetings and Scheduling

7.Projects

These labels provide a structured framework for analysis, enabling us to derive actionable insights and build robust machine learning models. Each of the 16,769 emails was assigned to one of these categories, ensuring that every email is associated with a clear and meaningful label. These labels not only reflect the content and context of the emails but also serve as the foundation for supervised learning tasks, such as classification and clustering.

### Why These Labels?

Personal: Emails related to individual or non-work-related communication.

Operations and Logistics: Emails concerning supply chain, logistics, and operational workflows.

Corporate and Legal: Emails involving legal matters, compliance, and corporate governance.

Finance: Emails related to budgeting, trading, and financial operations.

HR: Emails pertaining to recruitment, employee relations, and human resources management.

Meetings and Scheduling: Emails about meeting coordination, calendars, and scheduling.

Projects: Emails tied to specific initiatives, project management, and collaborative efforts.

By using these categories as labels, we ensure that our dataset is not only well-organized but also primed for advanced analytical tasks, such as predictive modeling and semantic analysis. This approach allows us to leverage the inherent structure of the data while maintaining its contextual richness.

## Features
1.Content Length – The total length of the email body.

2.Number of Recipients – The count of recipients in the email.

3.Email Subject Embedding – A numerical representation of the email subject.

4.Email Body Embedding – A numerical representation of the email body.

For the last two features, we experimented with different embedding techniques. Initially, we used Word2Vec, but the results were not satisfactory. To improve performance, we turned to BERT (Bidirectional Encoder Representations from Transformers). The difference was significant, as demonstrated in the figures.

# Hypothesis
Given that our data is not linearly separable and has high dimensionality due to the embeddings, simpler models like Logistic Regression may struggle to effectively classify the emails. These models often face difficulties when dealing with complex, high-dimensional feature spaces.

In contrast, more advanced algorithms, such as Random Forest, are better suited for handling large numbers of dimensions. Their ability to capture intricate patterns and relationships in the data makes them a more promising choice for our classification task. 


## Machine Learning Models Used

To classify emails effectively, we experimented with multiple models:

### Multi Class Logistic Regression
Applied to predict certain numerical properties of emails based on extracted features.

### Random Forest
Used for classification tasks due to its robustness against overfitting and high interpretability.

### Support Vector Machine
Used soft SVM as the soft SVM more robust to noisy data such our own.

## Visualizations

To analyze and interpret the dataset, we used:

Matplotlib for static visualizations.

Seaborn for enhanced scatter plots.

Bar charts to visualize folder distributions before and after data cleaning.

## Key Findings & Impact

add here how each performed with direction to the respected figure's

### Multi Class Logistic Regression

### Random Forest

### Support Vector Machine


We observed that Word2Vec did not provide satisfactory separation or model results for our email categorization task. Upon further research, we discovered that Word2Vec generates word vectors in a way that does not always preserve the semantic meaning of words, especially in contexts where word order and deeper contextual understanding are crucial.

To address this, we transitioned to using BERT (Bidirectional Encoder Representations from Transformers) for vectorizing the email text. Unlike Word2Vec, BERT captures the contextual meaning of words by considering the entire sentence structure, resulting in richer and more semantically meaningful embeddings.

Additionally, BERT generates higher-dimensional vectors, which provide a more nuanced representation of the text. This shift allowed us to better capture the semantic relationships between words and phrases in the emails, ultimately improving the performance of our categorization models.

## Future Improvements

### Diversify Data Sources
Expand Data Collection: Currently, we rely on the Ernon dataset, which primarily consists of business corporate data. To achieve a more comprehensive understanding, we should incorporate data from various sectors such as:

Universities: Academic and research data.

Hospitals: Healthcare and medical records.

Construction: Infrastructure and project-related data.

Other Fields: Include data from retail, technology, agriculture, and more.

Categorize Jobs: By integrating data from multiple sectors, we can better categorize and analyze job roles across different industries.

### Integrate Advanced Algorithms
Explore Complex Models: While we have utilized algorithms like Random Forest, DBSCAN, Logistic Regression (LG), and Support Vector Machines (SVM), there is potential to integrate more sophisticated models such as:

Gradient Boosting Machines (GBM): For improved predictive accuracy.

XGBoost/LightGBM: Efficient and scalable implementations of gradient boosting.

Neural Networks: Deep learning models for capturing complex patterns.

Ensemble Methods: Combine multiple models to enhance performance.

Hyperparameter Tuning: Optimize the parameters of these algorithms to achieve better results.

### Enhance Word Embeddings
Experiment with Embedding Techniques: Explore various word embedding algorithms to better capture semantic relationships in the data. Some options include:

GloVe: Global Vectors for word representation.

FastText: For capturing subword information.

Domain-Specific Embeddings: Train embeddings on domain-specific corpora to better fit the nuances of different sectors.

### Improve Data Quality and Preprocessing

Data Cleaning: Implement more rigorous data cleaning techniques to handle missing values, outliers, and inconsistencies.

Feature Engineering: Create more informative features that can enhance model performance.

Data Augmentation: Use techniques like SMOTE (Synthetic Minority Over-sampling Technique) to handle imbalanced datasets.




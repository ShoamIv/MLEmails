import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import email
from bs4 import BeautifulSoup


class EmailClassifier:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)  # Include bigrams for more context
        )
        self.classifier = MultiOutputClassifier(RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced'
        ))
        self.mlb = MultiLabelBinarizer()
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_email(self, raw_email):
        """
        Preprocess raw email content
        """
        # Handle different input types
        if isinstance(raw_email, dict):
            raw_email = f"{raw_email.get('subject', '')} {raw_email.get('body', '')}"

        # Parse email
        try:
            email_content = email.message_from_string(str(raw_email))
        except:
            # If parsing fails, use the raw input
            email_content = raw_email

        # Extract text from email
        body = ""
        if hasattr(email_content, 'is_multipart') and email_content.is_multipart():
            for part in email_content.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload()
        else:
            body = str(email_content)

        # Remove HTML tags
        soup = BeautifulSoup(body, 'html.parser')
        text = soup.get_text()

        # Basic preprocessing
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters

        # Tokenization and lemmatization
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

        return ' '.join(tokens)

    def fit(self, emails, categories):
        """
        Train the classifier on preprocessed emails and their categories
        """
        # Preprocess all emails
        processed_emails = [self.preprocess_email(email) for email in emails]

        # Convert text to TF-IDF features
        X = self.vectorizer.fit_transform(processed_emails)

        # Transform categories to binary matrix
        y = self.mlb.fit_transform(categories)

        # Train the classifier
        self.classifier.fit(X, y)

    def predict(self, emails):
        """
        Predict categories for one or more emails
        """
        # Handle single email input
        if isinstance(emails, str):
            emails = [emails]

        # Preprocess the emails
        processed_emails = [self.preprocess_email(email) for email in emails]

        # Transform to TF-IDF features
        X = self.vectorizer.transform(processed_emails)

        # Predict categories
        predictions = self.classifier.predict(X)

        # Convert binary predictions back to category labels
        categories = self.mlb.inverse_transform(predictions)

        return categories

    def evaluate(self, test_emails, true_categories):
        """
        Evaluate the classifier's performance
        """
        # Preprocess test emails
        processed_emails = [self.preprocess_email(email) for email in test_emails]
        X_test = self.vectorizer.transform(processed_emails)

        # Transform true categories to binary matrix
        y_test = self.mlb.transform(true_categories)

        # Get predictions
        y_pred = self.classifier.predict(X_test)

        # Generate detailed classification report
        print("\nDetailed Classification Report:")
        report = classification_report(
            y_test,
            y_pred,
            target_names=self.mlb.classes_,
            zero_division=1  # Replace undefined metrics with 1
        )
        print(report)

        # Additional evaluation metrics
        print("\nCross-Validation Scores (on training set):")
        cv_scores = cross_val_score(
            self.classifier,
            self.vectorizer.transform(processed_emails),
            y_test,
            cv=3,  # 3-fold cross-validation
            scoring='accuracy'
        )
        print(f"Cross-validation accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

        return report


# Example usage in main script
def main():
    # Load dataset
    df = pd.read_csv("email_dataset.csv")


    # Prepare email texts and categories
    email_texts = (df["subject"] + " " + df["body"]).tolist()
    categories = df["primary_category"].apply(lambda x: x.split(",")).tolist()  # Convert category strings to lists

    # Initialize and train classifier
    classifier = EmailClassifier()

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        email_texts, categories,
        test_size=0.2,
        random_state=42,
        stratify=categories
    )

    # Train the classifier
    print("Training the classifier...")
    classifier.fit(X_train, y_train)

    # Evaluate the model
    print("\nModel Evaluation:")
    classifier.evaluate(X_test, y_test)

    # Test prediction on a new email
    print("\nTesting Prediction:")
    test_email = """
    Subject: Team Collaboration Update
    Hello colleagues,
    We are planning a new collaborative project to improve our team's 
    productivity and communication. We'll be discussing strategies 
    and potential tools for better project management.
    Looking forward to your input!
    """
    predicted_categories = classifier.predict(test_email)
    print("Predicted Categories:", predicted_categories)


if __name__ == "__main__":
    main()
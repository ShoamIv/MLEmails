import random
import pandas as pd
import numpy as np
from faker import Faker


class EmailDatasetGenerator:
    def __init__(self, seed=42):
        self.fake = Faker()
        random.seed(seed)

        # Predefined email templates for different categories
        self.email_templates = {
            "work": [
                "Project {project_name} Update: {details}",
                "Team Meeting Agenda: {details}",
                "Quarterly Performance Review: {details}",
                "Collaboration Request: {details}"
            ],
            "finance": [
                "Monthly Billing Statement: {details}",
                "Expense Report Reminder: {details}",
                "Investment Portfolio Update: {details}",
                "Tax Documentation: {details}"
            ],
            "personal": [
                "Family Reunion Planning: {details}",
                "Vacation Invitation: {details}",
                "Birthday Party Invitation: {details}",
                "Personal Achievement Sharing: {details}"
            ],
            "marketing": [
                "Special Promotion: {details}",
                "New Product Launch: {details}",
                "Seasonal Sale Announcement: {details}",
                "Exclusive Offer for Subscribers: {details}"
            ],
            "newsletter": [
                "Tech Trends Monthly: {details}",
                "Industry Insights: {details}",
                "Scientific Breakthroughs: {details}",
                "Global News Digest: {details}"
            ],
            "travel": [
                "Upcoming Trip Itinerary: {details}",
                "Travel Recommendations: {details}",
                "Flight Booking Confirmation: {details}",
                "Hotel Reservation Details: {details}"
            ],
            "education": [
                "Course Registration: {details}",
                "Academic Workshop Invitation: {details}",
                "Research Opportunity: {details}",
                "Learning Resources Update: {details}"
            ]
        }

        # Possible secondary categories for multi-label classification
        self.secondary_categories = {
            "work": ["meeting", "project", "review", "collaboration"],
            "finance": ["billing", "investment", "expense", "tax"],
            "personal": ["family", "invitation", "celebration", "achievement"],
            "marketing": ["promotion", "sale", "product", "offer"],
            "newsletter": ["technology", "science", "news", "innovation"],
            "travel": ["itinerary", "booking", "recommendation", "destination"],
            "education": ["workshop", "course", "research", "resources"]
        }

    def generate_email_content(self, category):
        """Generate email content based on category"""
        template = random.choice(self.email_templates[category])

        # Generate realistic details
        details_generators = {
            "work": lambda: f"for {self.fake.company()} involving {self.fake.bs()}",
            "finance": lambda: f"totaling ${random.uniform(50, 5000):.2f}",
            "personal": lambda: f"with {self.fake.name()} on {self.fake.date()}",
            "marketing": lambda: f"for {self.fake.company_suffix()} product line",
            "newsletter": lambda: f"about {self.fake.catch_phrase()}",
            "travel": lambda: f"to {self.fake.country()} on {self.fake.date()}",
            "education": lambda: f"in {self.fake.bs()}"
        }

        details = details_generators[category]()

        return template.format(project_name=self.fake.word(), details=details)

    def generate_email_metadata(self, category):
        """Generate email metadata"""
        return {
            "sender": self.fake.email(),
            "recipient": self.fake.email(),
            "timestamp": self.fake.date_time(),
            "subject": self.generate_email_content(category),
            "body": self.fake.paragraph(nb_sentences=5),
            "primary_category": category,
            "secondary_categories": random.sample(
                self.secondary_categories[category],
                random.randint(1, 2)
            )
        }

    def generate_dataset(self, num_emails=100):
        """Generate a synthetic email dataset"""
        emails = []

        # Ensure balanced distribution of categories
        categories = list(self.email_templates.keys())
        emails_per_category = num_emails // len(categories)
        remainder = num_emails % len(categories)

        for category in categories:
            category_count = emails_per_category + (1 if remainder > 0 else 0)
            remainder -= 1

            for _ in range(category_count):
                email = self.generate_email_metadata(category)
                emails.append(email)

        return pd.DataFrame(emails)

    def save_to_csv(self, dataset, filename='email_dataset.csv'):
        """Save dataset to CSV"""
        dataset.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")


# Usage example
def main():
    # Create generator
    generator = EmailDatasetGenerator()

    # Generate dataset
    dataset = generator.generate_dataset(num_emails=100)

    # Display dataset info
    print("\nDataset Overview:")
    print(dataset['primary_category'].value_counts())

    # Save to CSV
    generator.save_to_csv(dataset)

    # Display first few rows
    print("\nFirst 5 Rows:")
    print(dataset[['sender', 'subject', 'primary_category', 'secondary_categories']].head())


if __name__ == "__main__":
    main()
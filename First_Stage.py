import email
import os
import pandas as pd
import matplotlib.pyplot as plt


def parse_email(raw_email):
    msg = email.message_from_string(raw_email)
    date = msg['Date']
    sender = msg['From']
    recipients = msg['To']
    subject = msg['Subject'] if msg['Subject'] else ""

    # Extract body
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_payload(decode=True).decode(errors='ignore')
    else:
        body = msg.get_payload(decode=True).decode(errors='ignore')

    content_length = len(body)
    num_recipients = len(recipients.split(",")) if recipients else 0
    x_folder = msg['X-Folder'].split("\\")[-1].lower() if 'X-Folder' in msg else "Unknown"

    return pd.Series([x_folder, date, sender, recipients, subject, body, content_length, num_recipients])


def initiate_stage1(datastore_path, figure_folder):
    # Load dataset
    emails = pd.read_csv(os.path.join(datastore_path, "emails.csv"))

    # Apply function to extract headers
    emails[['X-Folder', 'Date', 'From', 'To', 'Subject', 'Body', 'Content-Length', 'Number of Recipients']] = \
        emails['message'].apply(parse_email)

    # Drop unnecessary columns
    emails.drop(columns=['file', 'Date'], inplace=True, errors='ignore')

    # Count emails based on recipients
    one_recipient_count = (emails['Number of Recipients'] == 1).sum()
    multiple_recipients_count = (emails['Number of Recipients'] > 1).sum()
    print("Number of emails with 1 recipient:", one_recipient_count)
    print("Number of emails with more than 1 recipient:", multiple_recipients_count)

    # Count and plot folder occurrences
    folder_counts = emails['X-Folder'].value_counts().reset_index()
    folder_counts.columns = ['X-Folder', 'Count']
    plt.figure(figsize=(8, 10))
    plt.barh(folder_counts['X-Folder'][:40], folder_counts['Count'][:40], color='skyblue')
    plt.xlabel("Count")
    plt.ylabel("X-Folder")
    plt.title("Top 40 Email Folders by Count")
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(figure_folder, "email_Top_40_Folder.png"), bbox_inches="tight")

    # Remove generic folders
    generic_folders = {"inbox", "sent", "deleted items", "drafts", "all documents", "discussion threads",
                       "sent items", "untitled", "notes inbox", "calendar"}
    emails = emails[~emails['X-Folder'].isin(generic_folders)]

    # Merge similar folder names
    merge_folders = {"personnel": "personal", "personalfolder": "personal", "corporate": "corporate and legal",
                     "legal": "corporate and legal", "budget": "finance", "market": "finance",
                     "trading info": "finance", "internet": "it"}
    emails['X-Folder'] = emails['X-Folder'].replace(merge_folders)

    # Filter relevant folders
    folders_to_keep = ["personal", "hr", "meetings and scheduling", "operations and logistics", "projects",
                       "corporate and legal", "finance"]
    emails = emails[emails['X-Folder'].isin(folders_to_keep)]

    # Save the filtered emails
    filtered_path = os.path.join(datastore_path, "filtered_emails.csv")
    emails.to_csv(filtered_path, index=False)
    print("Filtered CSV saved as 'filtered_emails.csv'")

    # Final cleaning: remove empty subjects and archive-related folders
    emails = emails[emails['Subject'].notna() & (emails['Subject'] != "")]
    emails = emails[~emails['X-Folder'].isin(["it", "archive and miscellaneous"])]

    # Plot folder occurrences after cleaning
    folder_counts = emails['X-Folder'].value_counts().reset_index()
    folder_counts.columns = ['X-Folder', 'Count']
    plt.figure(figsize=(8, 10))
    plt.barh(folder_counts['X-Folder'], folder_counts['Count'], color='skyblue')
    plt.xlabel("Count")
    plt.ylabel("X-Folder")
    plt.title("Email Folders by Count After Cleanup")
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(figure_folder, "folder_occurrences_afterClean.png"), bbox_inches="tight")


def Run_FirstStage(datastore_path, figure_folder):
    initiate_stage1(datastore_path, figure_folder)

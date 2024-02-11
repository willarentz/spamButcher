# SpamButcher

SpamButcher is an intelligent (GPT3.5/GPT4/ollama) email filtering tool that connects to your IMAP server to help you manage your inbox more efficiently by identifying and handling spam emails. Using advanced spam LLMs, it reviews your unread emails, determines which ones are spam, and takes appropriate actions to keep your inbox clean.

## Key Features
- Automated Spam Detection: Utilizes a custom is_spam function to evaluate each email and identify unwanted spam.
- Email Management: Saves a copy of all processed emails to a local database for record-keeping and further analysis.
- Spam Isolation: Moves detected spam emails to a separate 'SpamAI' folder within your email server, decluttering your main inbox.
- Continuous Operation: Operates in a loop, rechecking the inbox every 10 minutes, ensuring that new emails are processed in a timely manner.

## How It Works
1. Connect: Establish a secure connection to your designated IMAP server.
2. Retrieve: Fetch all unread emails from your inbox.
3. Process: Analyze each email to determine if it is spam.
4. Save: Record emails to the database for archival purposes.
5. Organize: Transfer spam emails to 'SpamAI' folder.
6. Clean Up: Delete processed emails from the inbox.
7. Pause & Repeat: Wait for 10 minutes before starting the cycle again.

## Getting Started
To get started with SpamButcher, clone this repository and follow the setup instructions in the documentation to configure your IMAP server details and customize the spam detection settings according to your preferences.

```
git clone https://github.com/willarentz/spamButcher.git
cd spamButcher
python spamButcher.py
```

## Contributing
Any contributions you make are greatly appreciated.

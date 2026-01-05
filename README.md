# SpamButcher

SpamButcher is an intelligent (Ollama / OpenAI / Gemini) email filtering tool that connects to your IMAP server, continuously classifies new messages, and isolates spam into a dedicated folder. The new control center exposes a full monitoring UI plus live configuration so you can run the background worker without editing code.

## Key Features
- **Flexible LLM pipeline** – plug into a local Ollama model or hosted OpenAI / Gemini models by changing a dropdown.
- **Multi-account IMAP support** – onboard any number of inboxes, each with its own folders, and process them in a single run.
- **Automatic inbox hygiene** – unread emails are normalized, stored in SQLite, and copy/moved to your `SpamAI` folder when marked as spam.
- **Web control center** – live status cards, processed-email log with drill-down details, and an event feed surface everything happening in real time.
- **Inline configuration** – IMAP credentials, poll interval, and provider-specific settings are editable from the dashboard modal and persisted to `config.json`.
- **Manual review queue** – `/manual` still serves the JSON review table so you can build custom datasets when needed.

## How It Works
1. Background worker logs into your IMAP inbox, fetches unread messages, and normalizes headers + bodies.
2. A selected LLM receives the prompt and returns `{ "isSpam": true|false, "reason": "..." }`.
3. Results are committed to `emails.db`, surfaced in the UI, and spam copies are moved to the configured folder before the originals are deleted.
4. The worker sleeps for the configured poll interval (default 10 minutes) and repeats.

## Getting Started
```
git clone https://github.com/willarentz/spamButcher.git
cd spamButcher
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # create one if needed
export FLASK_APP=app.py
export SPAMBUTCHER_PORT=5150
flask run  # open http://localhost:5150
```

1. Open the dashboard, click **Edit Configuration**, and fill in IMAP + provider settings.
2. Start the worker or run a one-off cycle from the header buttons.
3. Use the processed email log to inspect classifications or click **Manual Review** to label JSON exports.

## Contributing
Any contributions you make are greatly appreciated.

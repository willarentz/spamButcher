# SpamButcher

SpamButcher is an intelligent (Ollama / OpenAI / Gemini) email filtering tool that connects to your IMAP server, continuously classifies new messages, and isolates spam into a dedicated folder. The new control center exposes a full monitoring UI plus live configuration so you can run the background worker without editing code.

## Key Features
- **Flexible LLM pipeline** – plug into a local Ollama model or hosted OpenAI / Gemini models by changing a dropdown.
- **Multi-account mailbox support** – onboard any number of IMAP or Google inboxes, each with its own folders, and process them in a single run.
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

## Connecting Google mailboxes
1. In [Google Cloud Console](https://console.cloud.google.com/), create an OAuth **Web application** client for Gmail, add `http://YOUR_HOST:YOUR_PORT/google/oauth/callback` (the UI shows the exact URL) to the **Authorized redirect URIs**, and download the JSON credentials file.
2. In the dashboard’s **Accounts & Polling** tab, choose **Google** as the account type, paste or upload the OAuth credential JSON, save, then click **Open Google Consent** to launch the built-in authorization popup. SpamButcher stores both the credentials and the refresh token inside `google_auth/` automatically, so you never have to juggle token files or paths.
3. Optionally customize the Gmail query (defaults to `label:INBOX is:unread`) and the target spam label. The worker will create the label if it does not exist and move spam out of the Inbox.
4. After you approve the consent screen once, the refresh token is saved and reused silently. If you prefer a console-based flow (for completely headless environments), set `SPAMBUTCHER_GOOGLE_AUTH_MODE=console` before running the worker.

Notes:
- Google requires HTTPS for OAuth redirect URIs unless you use `http://localhost`. If you open the UI on a private IP (for example, `http://192.168.x.x:5150`), OAuth will fail unless you proxy HTTPS.
- For local/private IP development, SpamButcher enables `OAUTHLIB_INSECURE_TRANSPORT=1` automatically. You can force it with `SPAMBUTCHER_ALLOW_INSECURE_OAUTH=1` if needed. Do not use insecure transport on public hosts.

## Contributing
Any contributions you make are greatly appreciated.

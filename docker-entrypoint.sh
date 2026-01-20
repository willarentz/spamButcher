#!/bin/sh
set -e

CONFIG_PATH="${SPAMBUTCHER_CONFIG_PATH:-/app/config.json}"

# Ensure data directory exists
mkdir -p "$(dirname "$CONFIG_PATH")"

# Create default config.json if it doesn't exist
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Creating default config at $CONFIG_PATH..."
    cat > "$CONFIG_PATH" << 'EOF'
{
    "accounts": [],
    "poll_interval_seconds": 600,
    "llm_provider": "ollama",
    "system_prompt": "You are a spam classifier. Analyze the email and respond with ONLY 'spam' or 'ham'. No explanations.",
    "ollama": {
        "base_url": "http://host.docker.internal:11434",
        "model": "llama3.2"
    },
    "openai": {
        "api_key": "",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini"
    },
    "gemini": {
        "api_key": "",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-1.5-flash"
    }
}
EOF
fi

# Ensure directories exist
mkdir -p /app/mails/ham /app/mails/spam /app/google_auth /app/results /app/data

exec "$@"

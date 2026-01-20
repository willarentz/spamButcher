FROM python:3.11-slim

LABEL maintainer="spamButcher"
LABEL description="AI-powered email spam filter using LLMs"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SPAMBUTCHER_PORT=5150

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY spam_processing.py .
COPY spamButcher.py .
COPY templates/ templates/
COPY static/ static/

# Copy and setup entrypoint
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Expose the web UI port
EXPOSE 5150

# Volume mount points for persistence
VOLUME ["/app/google_auth", "/app/mails"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5150/')" || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "app.py"]

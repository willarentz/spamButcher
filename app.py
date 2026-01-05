from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
import json
import os
import shutil
from copy import deepcopy

import requests

from spam_processing import (
    ConfigManager,
    DEFAULT_CONFIG,
    EmailDatabase,
    EmailProcessor,
    ProcessingMonitor,
    ProcessorWorker,
)


app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'spam-butcher-dev')

HAM_FOLDER = 'mails/ham/'
SPAM_FOLDER = 'mails/spam/'

os.makedirs('mails', exist_ok=True)
os.makedirs(HAM_FOLDER, exist_ok=True)
os.makedirs(SPAM_FOLDER, exist_ok=True)

config_manager = ConfigManager()
monitor = ProcessingMonitor()
database = EmailDatabase()
processor = EmailProcessor(config_manager, monitor, database)
worker = ProcessorWorker(processor, monitor, config_manager)


def read_emails_from_json():
    emails = []
    for filename in os.listdir('mails/'):
        if filename.endswith('.json'):
            with open(os.path.join('mails', filename), 'r', encoding='utf-8') as handle:
                json_file = json.load(handle)
                json_file['filename'] = filename
                if 'is_spam' in json_file:
                    json_file.pop('is_spam')
                emails.append(json_file)
    return emails


def _parse_int(value, default):
    if value is None or value == '':
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_accounts(form):
    count = _parse_int(form.get('accounts-count'), 0)
    accounts = []
    for idx in range(count):
        prefix = f'accounts-{idx}-'
        name = form.get(prefix + 'name', '').strip()
        imap_host = form.get(prefix + 'imap_host', '').strip()
        imap_username = form.get(prefix + 'imap_username', '').strip()
        imap_password = form.get(prefix + 'imap_password', '').strip()
        imap_folder = form.get(prefix + 'imap_folder', 'INBOX').strip() or 'INBOX'
        spam_folder = form.get(prefix + 'spam_folder', 'SpamAI').strip() or 'SpamAI'
        download_max = max(1, _parse_int(form.get(prefix + 'download_max'), DEFAULT_CONFIG['accounts'][0]['download_max']))
        enabled = form.get(prefix + 'enabled') == 'on'
        if not (imap_host and imap_username and imap_password):
            continue
        accounts.append(
            {
                'name': name or f'Account {len(accounts) + 1}',
                'imap_host': imap_host,
                'imap_username': imap_username,
                'imap_password': imap_password,
                'imap_folder': imap_folder,
                'spam_folder': spam_folder,
                'download_max': download_max,
                'enabled': enabled,
            }
        )
    if not accounts:
        accounts = deepcopy(DEFAULT_CONFIG['accounts'])
    return accounts


@app.route('/')
def dashboard():
    context = {
        'config': config_manager.get_config(),
        'status': monitor.get_status(),
        'emails': database.recent_emails(25),
        'worker_thread_running': worker.is_running(),
    }
    return render_template('dashboard.html', **context)


@app.route('/config', methods=['POST'])
def update_config():
    form = request.form
    accounts = _parse_accounts(form)
    updates = {
        'accounts': accounts,
        'poll_interval_seconds': _parse_int(
            form.get('poll_interval_seconds'),
            DEFAULT_CONFIG['poll_interval_seconds'],
        ),
        'llm_provider': form.get('llm_provider', 'ollama'),
        'system_prompt': form.get('system_prompt', DEFAULT_CONFIG['system_prompt']).strip()
        or DEFAULT_CONFIG['system_prompt'],
        'ollama': {
            'base_url': form.get('ollama_base_url', DEFAULT_CONFIG['ollama']['base_url']).strip()
            or DEFAULT_CONFIG['ollama']['base_url'],
            'model': form.get('ollama_model', DEFAULT_CONFIG['ollama']['model']).strip()
            or DEFAULT_CONFIG['ollama']['model'],
        },
        'openai': {
            'api_key': form.get('openai_api_key', '').strip(),
            'base_url': form.get('openai_base_url', DEFAULT_CONFIG['openai']['base_url']).strip()
            or DEFAULT_CONFIG['openai']['base_url'],
            'model': form.get('openai_model', DEFAULT_CONFIG['openai']['model']).strip()
            or DEFAULT_CONFIG['openai']['model'],
        },
        'gemini': {
            'api_key': form.get('gemini_api_key', '').strip(),
            'base_url': form.get('gemini_base_url', DEFAULT_CONFIG['gemini']['base_url']).strip()
            or DEFAULT_CONFIG['gemini']['base_url'],
            'model': form.get('gemini_model', DEFAULT_CONFIG['gemini']['model']).strip()
            or DEFAULT_CONFIG['gemini']['model'],
        },
    }
    config_manager.update_config(updates)
    flash('Configuration updated', 'success')
    return redirect(url_for('dashboard'))


@app.route('/worker/start', methods=['POST'])
def start_worker():
    worker.start()
    flash('Processing worker started', 'success')
    return redirect(url_for('dashboard'))


@app.route('/worker/stop', methods=['POST'])
def stop_worker():
    worker.stop()
    flash('Processing worker stopped', 'info')
    return redirect(url_for('dashboard'))


@app.route('/worker/run-once', methods=['POST'])
def run_once():
    if worker.is_running():
        flash('Stop the background worker before running a manual cycle', 'warning')
    else:
        worker.run_once()
        flash('Processing cycle queued', 'info')
    return redirect(url_for('dashboard'))


@app.route('/api/status')
def api_status():
    status = monitor.get_status()
    status['worker_thread_running'] = worker.is_running()
    status['poll_interval_seconds'] = config_manager.get_config().get('poll_interval_seconds', 600)
    return jsonify(status)


@app.route('/api/emails')
def api_emails():
    limit = _parse_int(request.args.get('limit', 25), 25)
    limit = max(1, min(limit, 200))
    return jsonify(database.recent_emails(limit))


@app.route('/api/ollama/models')
def api_ollama_models():
    config = config_manager.get_config()
    default_base = DEFAULT_CONFIG['ollama']['base_url']
    base_url = request.args.get('base_url') or config.get('ollama', {}).get('base_url') or default_base
    base_url = (base_url or default_base).rstrip('/')
    if not base_url:
        base_url = default_base.rstrip('/')
    url = f'{base_url}/api/tags'
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json() or {}
        models_data = data.get('models') or data.get('tags') or []
        models = []
        for item in models_data:
            if isinstance(item, str):
                models.append(item)
            elif isinstance(item, dict):
                name = item.get('name') or item.get('model')
                if name:
                    models.append(name)
        return jsonify({'models': models})
    except Exception as exc:
        return jsonify({'models': [], 'error': str(exc)}), 502


@app.route('/api/emails/<path:message_id>')
def api_email_detail(message_id):
    account_name = request.args.get('account')
    email_record = database.get_email(message_id, account_name)
    if not email_record:
        return jsonify({'error': 'Email not found'}), 404
    return jsonify(email_record)


@app.route('/emails/clear', methods=['POST'])
def clear_email_log():
    database.clear_emails()
    monitor.add_event('info', 'Email processing log cleared via UI')
    flash('Email processing log cleared', 'info')
    return redirect(url_for('dashboard'))


@app.route('/manual')
def manual_review():
    emails = read_emails_from_json()
    return render_template('emails.html', emails=emails)


@app.route('/classify', methods=['POST'])
def classify_email():
    email_file = request.form['email_file']
    classification = request.form['classification']

    source = os.path.join('mails', email_file)
    if classification == 'ham':
        shutil.move(source, os.path.join(HAM_FOLDER, email_file))
    elif classification == 'spam':
        shutil.move(source, os.path.join(SPAM_FOLDER, email_file))

    flash(f'Email {email_file} moved to {classification.upper()} folder', 'info')
    return redirect(url_for('manual_review'))


if __name__ == '__main__':
    port = int(os.environ.get('SPAMBUTCHER_PORT', '5150'))
    app.run(host='0.0.0.0', port=port, debug=True)

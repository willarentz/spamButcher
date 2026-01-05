from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
import json
import os
import shutil
import uuid
from copy import deepcopy
from ipaddress import ip_address
from typing import Dict, Optional

import requests
from google_auth_oauthlib.flow import Flow
from werkzeug.utils import secure_filename

from spam_processing import (
    ConfigManager,
    DEFAULT_CONFIG,
    EmailDatabase,
    EmailProcessor,
    ProcessingMonitor,
    ProcessorWorker,
    GOOGLE_SCOPES,
)


app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'spam-butcher-dev')

HAM_FOLDER = 'mails/ham/'
SPAM_FOLDER = 'mails/spam/'
GOOGLE_AUTH_DIR = 'google_auth'

os.makedirs('mails', exist_ok=True)
os.makedirs(HAM_FOLDER, exist_ok=True)
os.makedirs(SPAM_FOLDER, exist_ok=True)
os.makedirs(GOOGLE_AUTH_DIR, exist_ok=True)

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


def _enable_insecure_oauth_if_local() -> None:
    if os.environ.get('OAUTHLIB_INSECURE_TRANSPORT') == '1':
        return
    if os.environ.get('SPAMBUTCHER_ALLOW_INSECURE_OAUTH', '').lower() in {'1', 'true', 'yes'}:
        os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
        return
    host = (request.host or '').split(':', 1)[0]
    if host in {'localhost', '127.0.0.1', '::1'}:
        os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
        return
    try:
        if ip_address(host).is_private:
            os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    except ValueError:
        pass


def _parse_int(value, default):
    if value is None or value == '':
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _store_uploaded_json(file_storage, prefix):
    if not file_storage or not file_storage.filename:
        return None
    filename = secure_filename(file_storage.filename)
    if not filename.lower().endswith('.json'):
        return None
    safe_prefix = secure_filename(prefix) or 'google'
    unique_name = f"{safe_prefix}-{uuid.uuid4().hex}.json"
    path = os.path.join(GOOGLE_AUTH_DIR, unique_name)
    file_storage.save(path)
    return path


def _store_json_text(contents: str, prefix: str) -> Optional[str]:
    if not contents:
        return None
    try:
        parsed = json.loads(contents)
    except json.JSONDecodeError:
        return None
    safe_prefix = secure_filename(prefix) or 'google'
    unique_name = f"{safe_prefix}-{uuid.uuid4().hex}.json"
    path = os.path.join(GOOGLE_AUTH_DIR, unique_name)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(parsed, handle, indent=2)
    return path


def _generate_token_path(account_label: str) -> str:
    safe_label = secure_filename(account_label) or 'account'
    unique_suffix = uuid.uuid4().hex[:8]
    filename = f'{safe_label}-{unique_suffix}-token.json'
    return os.path.join(GOOGLE_AUTH_DIR, filename)


def _parse_accounts(form, files=None):
    count = _parse_int(form.get('accounts-count'), 0)
    accounts = []
    for idx in range(count):
        prefix = f'accounts-{idx}-'
        name = form.get(prefix + 'name', '').strip()
        account_label = secure_filename(name) or f'account-{idx + 1}'
        account_type = (form.get(prefix + 'type') or 'imap').strip().lower() or 'imap'
        imap_host = form.get(prefix + 'imap_host', '').strip()
        imap_username = form.get(prefix + 'imap_username', '').strip()
        imap_password = form.get(prefix + 'imap_password', '').strip()
        imap_folder = form.get(prefix + 'imap_folder', 'INBOX').strip() or 'INBOX'
        spam_folder = form.get(prefix + 'spam_folder', 'SpamAI').strip() or 'SpamAI'
        download_max = max(1, _parse_int(form.get(prefix + 'download_max'), DEFAULT_CONFIG['accounts'][0]['download_max']))
        google_credentials_file = form.get(prefix + 'google_credentials_file', '').strip()
        google_token_file = form.get(prefix + 'google_token_file', '').strip()
        google_credentials_data = form.get(prefix + 'google_credentials_data', '').strip()
        google_query = form.get(prefix + 'google_query', DEFAULT_CONFIG['accounts'][0]['google_query']).strip() or DEFAULT_CONFIG['accounts'][0]['google_query']
        enabled = form.get(prefix + 'enabled') == 'on'
        credentials_updated = False
        if files:
            creds_upload = files.get(prefix + 'google_credentials_upload')
            saved_creds = _store_uploaded_json(creds_upload, f'{account_label}-creds') if creds_upload else None
            if saved_creds:
                google_credentials_file = saved_creds
                credentials_updated = True
        if google_credentials_data:
            saved_from_text = _store_json_text(google_credentials_data, f'{account_label}-creds')
            if saved_from_text:
                google_credentials_file = saved_from_text
                credentials_updated = True
        if account_type == 'google':
            if not google_credentials_file:
                continue
            if credentials_updated or not google_token_file:
                google_token_file = _generate_token_path(account_label or f'account-{idx + 1}')
        elif not (imap_host and imap_username and imap_password):
            continue
        accounts.append(
            {
                'name': name or f'Account {len(accounts) + 1}',
                'type': account_type,
                'imap_host': imap_host,
                'imap_username': imap_username,
                'imap_password': imap_password,
                'imap_folder': imap_folder,
                'spam_folder': spam_folder,
                'download_max': download_max,
                'google_credentials_file': google_credentials_file,
                'google_token_file': google_token_file,
                'google_query': google_query,
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
        'google_callback_url': url_for('google_oauth_callback', _external=True),
    }
    return render_template('dashboard.html', **context)


@app.route('/config', methods=['POST'])
def update_config():
    form = request.form
    accounts = _parse_accounts(form, request.files)
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


def _resolve_account(account_index: int) -> Dict[str, Optional[Dict[str, Optional[str]]]]:
    config = config_manager.get_config()
    accounts = config.get('accounts') or []
    if 0 <= account_index < len(accounts):
        return {'config': config, 'accounts': accounts, 'account': accounts[account_index]}
    return {'config': config, 'accounts': accounts, 'account': None}


def _store_oauth_state(state: str, account_index: int) -> None:
    oauth_states = session.get('google_oauth_state')
    if not isinstance(oauth_states, dict):
        oauth_states = {}
    oauth_states[state] = {'account_index': account_index}
    session['google_oauth_state'] = oauth_states
    session.modified = True


def _pop_oauth_state(state: Optional[str]) -> Optional[Dict[str, int]]:
    if not state:
        return None
    oauth_states = session.get('google_oauth_state')
    if not isinstance(oauth_states, dict):
        return None
    entry = oauth_states.pop(state, None)
    session['google_oauth_state'] = oauth_states
    session.modified = True
    return entry


def _render_oauth_message(success: bool, message: str):
    return render_template('oauth_result.html', success=success, message=message)


@app.route('/google/oauth/start/<int:account_index>')
def google_oauth_start(account_index: int):
    context = _resolve_account(account_index)
    account = context['account']
    if not account:
        return _render_oauth_message(False, 'Selected account could not be found. Refresh the dashboard and try again.')
    if (account.get('type') or 'imap').lower() != 'google':
        return _render_oauth_message(False, 'This account is not configured as a Google mailbox.')
    credentials_path = account.get('google_credentials_file')
    if not credentials_path or not os.path.exists(credentials_path):
        return _render_oauth_message(False, 'Upload or paste the OAuth credential JSON, save the configuration, then start authorization.')
    _enable_insecure_oauth_if_local()
    redirect_uri = url_for('google_oauth_callback', _external=True)
    flow = Flow.from_client_secrets_file(credentials_path, scopes=GOOGLE_SCOPES, redirect_uri=redirect_uri)
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent',
    )
    _store_oauth_state(state, account_index)
    return redirect(authorization_url)


@app.route('/google/oauth/callback')
def google_oauth_callback():
    if request.args.get('error'):
        description = request.args.get('error_description') or 'Authorization was cancelled.'
        _pop_oauth_state(request.args.get('state'))
        return _render_oauth_message(False, description)
    _enable_insecure_oauth_if_local()
    state_entry = _pop_oauth_state(request.args.get('state'))
    if not state_entry:
        return _render_oauth_message(False, 'Authorization session expired. Please restart the Google consent flow.')
    account_index = state_entry.get('account_index', -1)
    context = _resolve_account(account_index)
    account = context['account']
    if not account:
        return _render_oauth_message(False, 'The selected mailbox was removed. Close this window and refresh the dashboard.')
    credentials_path = account.get('google_credentials_file')
    if not credentials_path or not os.path.exists(credentials_path):
        return _render_oauth_message(False, 'Credential file missing. Re-upload the OAuth JSON and try again.')
    redirect_uri = url_for('google_oauth_callback', _external=True)
    try:
        flow = Flow.from_client_secrets_file(
            credentials_path,
            scopes=GOOGLE_SCOPES,
            redirect_uri=redirect_uri,
            state=request.args.get('state'),
        )
        flow.fetch_token(authorization_response=request.url)
    except Exception as exc:  # pragma: no cover - network interaction
        return _render_oauth_message(False, f'Unable to complete authorization: {exc}')
    token_path = account.get('google_token_file')
    if not token_path:
        updated_accounts = context['config'].get('accounts', [])
        token_path = _generate_token_path(account.get('name') or f'Account {account_index + 1}')
        updated_accounts[account_index]['google_token_file'] = token_path
        config_manager.update_config({'accounts': updated_accounts})
    token_dir = os.path.dirname(token_path)
    if token_dir:
        os.makedirs(token_dir, exist_ok=True)
    with open(token_path, 'w', encoding='utf-8') as token_file:
        token_file.write(flow.credentials.to_json())
    return _render_oauth_message(True, 'Authorization complete. You can close this window.')


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


@app.route('/api/emails/<path:message_id>/reclassify', methods=['POST'])
def api_reclassify_email(message_id):
    payload = request.get_json(silent=True) or {}
    account_name = payload.get('account') or request.args.get('account')
    try:
        updated = processor.reclassify_email(message_id, account_name)
    except KeyError:
        return jsonify({'error': 'Email not found'}), 404
    except Exception as exc:
        monitor.add_event(
            'error',
            f'Re-evaluation failed for {message_id}: {exc}',
            {'account': account_name} if account_name else {},
        )
        return jsonify({'error': str(exc)}), 500
    return jsonify(updated)


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

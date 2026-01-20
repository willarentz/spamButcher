import base64
import email
import imaplib
import json
import os
import re
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime
from email.header import decode_header
from typing import Any, Dict, List, Optional

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request as GoogleRequest
    from googleapiclient.discovery import build as google_build
except ImportError:  # optional dependency, only needed for Google accounts
    Credentials = None
    InstalledAppFlow = None
    GoogleRequest = None
    google_build = None

import requests
from bs4 import BeautifulSoup


DATABASE_PATH = os.environ.get('SPAMBUTCHER_DB_PATH', 'emails.db')
CONFIG_PATH = os.environ.get('SPAMBUTCHER_CONFIG_PATH', 'config.json')


DEFAULT_CONFIG: Dict[str, Any] = {
    "accounts": [
        {
            "name": "Primary",
            "type": "imap",
            "imap_host": "",
            "imap_username": "",
            "imap_password": "",
            "imap_folder": "INBOX",
            "spam_folder": "SpamAI",
            "download_max": 50,
            "enabled": True,
            "google_credentials_file": "",
            "google_token_file": "",
            "google_query": "label:INBOX is:unread",
        }
    ],
    "poll_interval_seconds": 600,
    "llm_provider": "ollama",
    "system_prompt": (
        "You are SpamButcher, a careful email triage specialist. "
        "Classify each message as spam or ham with high precision. "
        "Respond ONLY with compact JSON like {\"isSpam\": true|false,"
        " \"reason\": \"...\"}."
    ),
    "ollama": {
        "base_url": "http://127.0.0.1:11434",
        "model": "spamfilter",
    },
    "openai": {
        "api_key": "",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "gemini": {
        "api_key": "",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-1.5-flash",
    },
}

GOOGLE_SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_merge(base[key], value)
        else:
            base[key] = value
    return base


class ConfigManager:
    def __init__(self, path: str = CONFIG_PATH) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._config = self._load()

    def _normalize_account(self, account: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        defaults = json.loads(json.dumps(DEFAULT_CONFIG['accounts'][0]))
        if account:
            defaults.update(account)
        defaults['type'] = (defaults.get('type') or 'imap').lower()
        defaults['imap_folder'] = defaults.get('imap_folder') or 'INBOX'
        defaults['spam_folder'] = defaults.get('spam_folder') or 'SpamAI'
        defaults['google_credentials_file'] = defaults.get('google_credentials_file') or ''
        defaults['google_token_file'] = defaults.get('google_token_file') or ''
        defaults['google_query'] = defaults.get('google_query') or DEFAULT_CONFIG['accounts'][0]['google_query']
        defaults['download_max'] = int(defaults.get('download_max', 50) or 50)
        defaults['enabled'] = bool(defaults.get('enabled', True))
        return defaults

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
        else:
            data = {}
        config = deep_merge(json.loads(json.dumps(DEFAULT_CONFIG)), data)
        legacy_fields = ['imap_host', 'imap_username', 'imap_password']
        if any(field in data for field in legacy_fields) and not data.get('accounts'):
            account = {
                'name': data.get('account_name') or 'Primary',
                'imap_host': data.get('imap_host', ''),
                'imap_username': data.get('imap_username', ''),
                'imap_password': data.get('imap_password', ''),
                'imap_folder': data.get('imap_folder', 'INBOX'),
                'spam_folder': data.get('spam_folder', 'SpamAI'),
                'download_max': int(data.get('download_max', 50) or 50),
                'enabled': True,
            }
            config['accounts'] = [account]
        if not config.get('accounts'):
            config['accounts'] = json.loads(json.dumps(DEFAULT_CONFIG['accounts']))
        config['accounts'] = [self._normalize_account(account) for account in config['accounts']]
        self._write(config)
        return config

    def _write(self, config: Dict[str, Any]) -> None:
        with open(self.path, 'w', encoding='utf-8') as handle:
            json.dump(config, handle, indent=2)

    def get_config(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._config))

    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            merged = deep_merge(self._config.copy(), updates)
            self._config = merged
            self._write(merged)
            return json.loads(json.dumps(self._config))


class ProcessingMonitor:
    def __init__(self, history_size: int = 200) -> None:
        self._history = deque(maxlen=history_size)
        self._lock = threading.Lock()
        self._worker_running = False
        self._last_run_start: Optional[datetime] = None
        self._last_run_finish: Optional[datetime] = None
        self._last_error: Optional[str] = None
        self._processed_total = 0

    def mark_run_start(self) -> None:
        with self._lock:
            self._worker_running = True
            self._last_run_start = datetime.utcnow()
        self.add_event('info', 'Processing cycle started')

    def mark_run_end(self, processed: int) -> None:
        with self._lock:
            self._worker_running = False
            self._last_run_finish = datetime.utcnow()
            self._processed_total += processed
        self.add_event('info', f'Processing cycle finished ({processed} emails)')

    def add_event(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': level,
            'message': message,
            'extra': extra or {},
        }
        with self._lock:
            self._history.appendleft(payload)
            if level == 'error':
                self._last_error = message

    def mark_error(self, message: str) -> None:
        self.add_event('error', message)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'worker_running': self._worker_running,
                'last_run_start': self._last_run_start.isoformat() + 'Z'
                if self._last_run_start
                else None,
                'last_run_finish': self._last_run_finish.isoformat() + 'Z'
                if self._last_run_finish
                else None,
                'last_error': self._last_error,
                'processed_total': self._processed_total,
                'events': list(self._history),
            }


class EmailDatabase:
    def __init__(self, path: str = DATABASE_PATH) -> None:
        self.path = path
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        conn = self._connect()
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='emails'")
        if cur.fetchone() is None:
            conn.execute(
                '''CREATE TABLE emails
                   (id TEXT, account_name TEXT, subject TEXT, date TEXT, from_email TEXT,
                    to_email TEXT, html_body TEXT, plain_text_body TEXT, is_spam BOOLEAN,
                    PRIMARY KEY (id, account_name))'''
            )
        else:
            columns = [row[1] for row in conn.execute('PRAGMA table_info(emails)')]
            if columns and ('account_name' not in columns or len(columns) != 9):
                conn.execute('ALTER TABLE emails RENAME TO emails_old')
                conn.execute(
                    '''CREATE TABLE emails
                       (id TEXT, account_name TEXT, subject TEXT, date TEXT, from_email TEXT,
                        to_email TEXT, html_body TEXT, plain_text_body TEXT, is_spam BOOLEAN,
                        PRIMARY KEY (id, account_name))'''
                )
                if 'account_name' in columns:
                    account_expr = 'account_name'
                else:
                    account_expr = "'Primary'"
                conn.execute(
                    '''INSERT OR REPLACE INTO emails
                       (id, account_name, subject, date, from_email, to_email, html_body, plain_text_body, is_spam)
                       SELECT id, ''' + account_expr + ''', subject, date, from_email, to_email, html_body, plain_text_body, is_spam
                       FROM emails_old'''
                )
                conn.execute('DROP TABLE emails_old')
        conn.commit()
        conn.close()

    def email_exists(self, message_id: str, account_name: str) -> bool:
        conn = self._connect()
        cur = conn.execute('SELECT 1 FROM emails WHERE id=? AND account_name=?', (message_id, account_name))
        row = cur.fetchone()
        conn.close()
        return row is not None

    def insert_email(self, email_payload: Dict[str, Any], account_name: str) -> None:
        conn = self._connect()
        conn.execute(
            'INSERT OR IGNORE INTO emails VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                email_payload['id'],
                account_name,
                email_payload.get('subject'),
                email_payload.get('date'),
                email_payload.get('from'),
                email_payload.get('to'),
                email_payload.get('body_html'),
                email_payload.get('body_plain_text'),
                int(bool(email_payload.get('isSpam'))),
            ),
        )
        conn.commit()
        conn.close()

    def recent_emails(self, limit: int = 25) -> List[Dict[str, Any]]:
        conn = self._connect()
        cur = conn.execute(
            'SELECT id, account_name, subject, date, from_email, to_email, is_spam FROM emails '
            'ORDER BY rowid DESC LIMIT ?',
            (limit,),
        )
        rows = cur.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def clear_emails(self) -> None:
        conn = self._connect()
        conn.execute('DELETE FROM emails')
        conn.commit()
        conn.close()

    def get_email(self, message_id: str, account_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        if account_name:
            cur = conn.execute(
                'SELECT * FROM emails WHERE id=? AND account_name=?',
                (message_id, account_name),
            )
        else:
            cur = conn.execute(
                'SELECT * FROM emails WHERE id=? ORDER BY rowid DESC LIMIT 1',
                (message_id,),
            )
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return dict(row)

    def update_classification(self, message_id: str, account_name: str, is_spam: bool) -> None:
        conn = self._connect()
        conn.execute(
            'UPDATE emails SET is_spam=? WHERE id=? AND account_name=?',
            (int(bool(is_spam)), message_id, account_name),
        )
        conn.commit()
        conn.close()


def find_body_in_email(email_message: email.message.Message) -> Dict[str, Optional[str]]:
    if not email_message.is_multipart():
        content_type = email_message.get_content_type()
        payload = email_message.get_payload(decode=True)
        charset = email_message.get_content_charset('utf-8')
        if payload is None:
            return {'html_body': None, 'plain_text_body': None}
        text = payload.decode(charset, errors='ignore')
        if content_type == 'text/html':
            return {'html_body': extract_text_from_html(text), 'plain_text_body': None}
        if content_type == 'text/plain':
            return {'html_body': None, 'plain_text_body': text}
        return {'html_body': None, 'plain_text_body': None}

    html_body = None
    plain_text_body = None
    for part in email_message.walk():
        if part.get_content_maintype() == 'multipart':
            continue
        content_type = part.get_content_type()
        charset = part.get_content_charset('utf-8')
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        text = payload.decode(charset, errors='ignore')
        if content_type == 'text/html' and not html_body:
            html_body = extract_text_from_html(text)
        elif content_type == 'text/plain' and not plain_text_body:
            plain_text_body = text
    return {'html_body': html_body, 'plain_text_body': plain_text_body}


def extract_text_from_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator=' ', strip=True)


def formatting_function(message: Dict[str, Any]) -> str:
    def safe(value: Optional[str]) -> str:
        return value or ''

    out = (
        f"TO: {safe(message.get('to'))}\n"
        f"FROM: {safe(message.get('from'))}\n"
        f"SUBJECT: {safe(message.get('subject'))}\n\n"
    )
    plain = safe(message.get('body_plain_text'))
    html = safe(message.get('body_html'))
    body = plain if len(plain) >= len(html) else html
    body = re.sub(r'[ \t]+', ' ', body)
    return out + body + '\n'


def extract_is_spam_substring(payload: str) -> Optional[str]:
    pattern = r'\{.*?"isSpam".*?\}'
    match = re.search(pattern, payload, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0)
    return None


class LLMClient:
    def generate(self, prompt: str, system_prompt: str) -> str:
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip('/')
        self.model = model

    def generate(self, prompt: str, system_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        response = requests.post(
            f'{self.base_url}/api/chat',
            json={'model': self.model, 'messages': messages, 'stream': False},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        message = data.get('message') or {}
        content = message.get('content', '')
        if not content and data.get('error'):
            raise RuntimeError(data['error'])
        return content


class OpenAIClient(LLMClient):
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, system_prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError('Missing OpenAI API key')
        conversation = []
        if system_prompt:
            conversation.append({'role': 'system', 'content': system_prompt})
        conversation.append({'role': 'user', 'content': prompt})
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        try:
            response = requests.post(
                f'{self.base_url}/responses',
                headers=headers,
                json={'model': self.model, 'input': conversation},
                timeout=120,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            raise RuntimeError(f'OpenAI request failed: {detail}') from exc
        data = response.json()
        text_segments = []
        for item in data.get('output', []) or []:
            for content in item.get('content', []) or []:
                text = content.get('text') or content.get('output_text')
                if text:
                    text_segments.append(text)
        if not text_segments:
            text_segments.extend(data.get('output_text') or [])
        if not text_segments:
            raise RuntimeError('OpenAI response missing output text')
        return '\n'.join(text_segments).strip()


class GeminiClient(LLMClient):
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, system_prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError('Missing Gemini API key')
        combined_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        url = f'{self.base_url}/models/{self.model}:generateContent?key={self.api_key}'
        payload = {
            'contents': [
                {
                    'role': 'user',
                    'parts': [{'text': combined_prompt}],
                }
            ]
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        candidates = data.get('candidates') or []
        if not candidates:
            raise RuntimeError('No completion returned from Gemini API')
        parts = candidates[0].get('content', {}).get('parts') or []
        if not parts:
            raise RuntimeError('Gemini response missing content parts')
        return parts[0].get('text', '')


def build_llm_client(config: Dict[str, Any]) -> LLMClient:
    provider = (config.get('llm_provider') or 'ollama').lower()
    if provider == 'openai':
        openai_cfg = config.get('openai', {})
        return OpenAIClient(
            base_url=openai_cfg.get('base_url', DEFAULT_CONFIG['openai']['base_url']),
            api_key=openai_cfg.get('api_key', ''),
            model=openai_cfg.get('model', DEFAULT_CONFIG['openai']['model']),
        )
    if provider == 'gemini':
        gemini_cfg = config.get('gemini', {})
        return GeminiClient(
            base_url=gemini_cfg.get('base_url', DEFAULT_CONFIG['gemini']['base_url']),
            api_key=gemini_cfg.get('api_key', ''),
            model=gemini_cfg.get('model', DEFAULT_CONFIG['gemini']['model']),
        )
    ollama_cfg = config.get('ollama', {})
    return OllamaClient(
        base_url=ollama_cfg.get('base_url', DEFAULT_CONFIG['ollama']['base_url']),
        model=ollama_cfg.get('model', DEFAULT_CONFIG['ollama']['model']),
    )


class EmailProcessor:
    def __init__(
        self,
        config_manager: ConfigManager,
        monitor: ProcessingMonitor,
        database: EmailDatabase,
    ) -> None:
        self.config_manager = config_manager
        self.monitor = monitor
        self.database = database
        self._gmail_label_cache: Dict[str, Dict[str, str]] = {}

    def process_new_messages(self) -> int:
        config = self.config_manager.get_config()
        accounts = [account for account in config.get('accounts', []) if account]
        if not accounts:
            self.monitor.add_event('warning', 'No IMAP accounts configured')
            return 0
        active_accounts = [account for account in accounts if account.get('enabled', True)]
        if not active_accounts:
            self.monitor.add_event('warning', 'All IMAP accounts are disabled')
            return 0
        total_processed = 0
        for account in active_accounts:
            total_processed += self._process_account(account, config)
        return total_processed

    def _process_account(self, account: Dict[str, Any], global_config: Dict[str, Any]) -> int:
        account_name = account.get('name') or account.get('imap_username') or 'Account'
        account_type = (account.get('type') or 'imap').lower()
        if account_type == 'google':
            return self._process_google_account(account_name, account, global_config)
        return self._process_imap_account(account_name, account, global_config)

    def reclassify_email(self, message_id: str, account_name: Optional[str] = None) -> Dict[str, Any]:
        config = self.config_manager.get_config()
        record = self.database.get_email(message_id, account_name)
        if not record:
            raise KeyError('Email not found')
        prompt_payload = {
            'to': record.get('to_email'),
            'from': record.get('from_email'),
            'subject': record.get('subject'),
            'body_plain_text': record.get('plain_text_body'),
            'body_html': record.get('html_body'),
        }
        client = build_llm_client(config)
        raw = client.generate(formatting_function(prompt_payload), config.get('system_prompt', ''))
        classification = self._interpret_response(raw)
        previous = bool(record.get('is_spam'))
        updated_is_spam = bool(classification['is_spam'])
        account_label = record.get('account_name') or account_name
        if not account_label:
            raise RuntimeError('Missing account label for re-evaluation')
        self.database.update_classification(message_id, account_label, updated_is_spam)
        level = 'warning' if updated_is_spam else 'info'
        outcome = 'changed' if previous != updated_is_spam else 'confirmed'
        subject = record.get('subject') or '(no subject)'
        self.monitor.add_event(
            level,
            f"[{account_label}] Re-evaluated email '{subject}' ({outcome}) as "
            f"{'spam' if updated_is_spam else 'ham'}",
            {'reason': classification.get('reason', ''), 'account': account_label},
        )
        record['is_spam'] = updated_is_spam
        record['previous_is_spam'] = previous
        record['reason'] = classification.get('reason', '')
        return record

    def _process_imap_account(self, account_name: str, account: Dict[str, Any], global_config: Dict[str, Any]) -> int:
        required = [
            field
            for field in ('imap_host', 'imap_username', 'imap_password')
            if not account.get(field)
        ]
        if required:
            self.monitor.add_event(
                'warning',
                f"[{account_name}] IMAP settings incomplete: {', '.join(required)}",
                {'account': account_name},
            )
            return 0

        download_max = int(account.get('download_max', 50) or 50)
        mail = None
        processed = 0
        try:
            try:
                mail = imaplib.IMAP4_SSL(account['imap_host'])
                mail.login(account['imap_username'], account['imap_password'])
            except OSError as exc:
                self.monitor.add_event(
                    'error',
                    f'[{account_name}] Unable to connect to IMAP host {account["imap_host"]}: {exc}',
                    {'account': account_name},
                )
                return 0
            except imaplib.IMAP4.error as exc:
                self.monitor.add_event(
                    'error',
                    f'[{account_name}] IMAP authentication failed: {exc}',
                    {'account': account_name},
                )
                return 0

            folder = account.get('imap_folder', 'INBOX')
            status, _ = mail.select(folder)
            if status != 'OK':
                raise RuntimeError(f'Unable to select folder {folder}')
            unread = self._fetch_unread_messages(mail, download_max)
            if not unread:
                self.monitor.add_event('info', f'[{account_name}] No unread emails found', {'account': account_name})
                return 0
            client = build_llm_client(global_config)
            for message in unread:
                if self.database.email_exists(message['id'], account_name):
                    continue
                prompt = formatting_function(message)
                try:
                    raw = client.generate(prompt, global_config.get('system_prompt', ''))
                    classification = self._interpret_response(raw)
                except Exception as exc:
                    self.monitor.add_event(
                        'error',
                        f"[{account_name}] Failed to classify email {message['subject']}: {exc}",
                        {'account': account_name},
                    )
                    continue
                message['isSpam'] = classification['is_spam']
                self.database.insert_email(message, account_name)
                processed += 1
                level = 'warning' if classification['is_spam'] else 'info'
                self.monitor.add_event(
                    level,
                    f"[{account_name}] Email '{message['subject']}' flagged as "
                    f"{'spam' if classification['is_spam'] else 'ham'}",
                    {'reason': classification.get('reason', ''), 'account': account_name},
                )
                if classification['is_spam']:
                    self._move_to_spam(
                        mail,
                        message['e_id'],
                        account.get('spam_folder', 'SpamAI'),
                        account_name,
                    )
            return processed
        except Exception as exc:
            self.monitor.add_event(
                'error',
                f'[{account_name}] Processing failed: {exc}',
                {'account': account_name},
            )
            return processed
        finally:
            if mail is not None:
                try:
                    mail.expunge()
                except Exception:
                    pass
                try:
                    mail.logout()
                except Exception:
                    pass

    def _process_google_account(self, account_name: str, account: Dict[str, Any], global_config: Dict[str, Any]) -> int:
        required = [
            field
            for field in ('google_credentials_file', 'google_token_file')
            if not account.get(field)
        ]
        if required:
            self.monitor.add_event(
                'warning',
                f"[{account_name}] Google settings incomplete: {', '.join(required)}",
                {'account': account_name},
            )
            return 0
        download_max = int(account.get('download_max', 50) or 50)
        query = account.get('google_query') or DEFAULT_CONFIG['accounts'][0]['google_query']
        processed = 0
        try:
            service = self._build_gmail_service(
                account['google_credentials_file'],
                account['google_token_file'],
            )
        except Exception as exc:
            self.monitor.add_event(
                'error',
                f'[{account_name}] Google authentication failed: {exc}',
                {'account': account_name},
            )
            return 0
        try:
            unread = self._fetch_google_unread_messages(service, query, download_max)
        except Exception as exc:
            self.monitor.add_event(
                'error',
                f'[{account_name}] Unable to fetch Google emails: {exc}',
                {'account': account_name},
            )
            return 0
        if not unread:
            self.monitor.add_event('info', f'[{account_name}] No unread Google emails found', {'account': account_name})
            return 0
        client = build_llm_client(global_config)
        for message in unread:
            if self.database.email_exists(message['id'], account_name):
                continue
            prompt = formatting_function(message)
            try:
                raw = client.generate(prompt, global_config.get('system_prompt', ''))
                classification = self._interpret_response(raw)
            except Exception as exc:
                self.monitor.add_event(
                    'error',
                    f"[{account_name}] Failed to classify email {message['subject']}: {exc}",
                    {'account': account_name},
                )
                continue
            message['isSpam'] = classification['is_spam']
            self.database.insert_email(message, account_name)
            processed += 1
            level = 'warning' if classification['is_spam'] else 'info'
            self.monitor.add_event(
                level,
                f"[{account_name}] Email '{message['subject']}' flagged as "
                f"{'spam' if classification['is_spam'] else 'ham'}",
                {'reason': classification.get('reason', ''), 'account': account_name},
            )
            if classification['is_spam']:
                spam_label = account.get('spam_folder', 'SpamAI') or 'SpamAI'
                try:
                    self._move_google_message_to_label(service, message['gmail_id'], spam_label, account_name)
                except Exception as exc:
                    self.monitor.add_event(
                        'error',
                        f'[{account_name}] Unable to move spam email: {exc}',
                        {'account': account_name},
                    )
        return processed

    def _fetch_unread_messages(self, mail: imaplib.IMAP4_SSL, download_max: int) -> List[Dict[str, Any]]:
        status, response = mail.search(None, '(UNSEEN)')
        if status != 'OK':
            return []
        msg_nums = response[0].split()
        if not msg_nums:
            return []
        msg_nums.reverse()
        emails: List[Dict[str, Any]] = []
        for e_id in msg_nums:
            status, data = mail.fetch(e_id, '(RFC822)')
            if status != 'OK':
                continue
            for part in data:
                if not isinstance(part, tuple):
                    continue
                msg = email.message_from_bytes(part[1])
                body = find_body_in_email(msg)
                emails.append(
                    {
                        'e_id': e_id,
                        'id': msg.get('Message-ID') or msg.get('message-id') or str(e_id.decode()),
                        'from': msg.get('from'),
                        'to': msg.get('to'),
                        'date': msg.get('date'),
                        'subject': self._decode_subject(msg.get('subject')),
                        'body_plain_text': body['plain_text_body'],
                        'body_html': body['html_body'],
                    }
                )
            if len(emails) >= download_max:
                break
        return emails

    def _decode_subject(self, subject_header: Optional[str]) -> str:
        if not subject_header:
            return ''
        decoded = decode_header(subject_header)
        parts = []
        for value, encoding in decoded:
            if isinstance(value, bytes):
                encoding = encoding or 'utf-8'
                parts.append(value.decode(encoding, errors='ignore'))
            else:
                parts.append(value)
        return ''.join(parts)

    def _interpret_response(self, content: str) -> Dict[str, Any]:
        snippet = extract_is_spam_substring(content)
        if not snippet:
            raise ValueError('Model response missing isSpam payload')
        data = json.loads(snippet)
        is_spam = bool(data.get('isSpam') or data.get('is_spam'))
        return {'is_spam': is_spam, 'reason': data.get('reason', '')}

    def _move_to_spam(self, mail: imaplib.IMAP4_SSL, e_id: bytes, folder: str, account_name: str) -> None:
        try:
            status, _ = mail.copy(e_id, folder)
            if status != 'OK':
                raise RuntimeError('copy failed')
            mail.store(e_id, '+FLAGS', '\\Deleted')
            self.monitor.add_event('info', f'[{account_name}] Moved email to spam folder', {'folder': folder, 'account': account_name})
        except Exception as exc:
            self.monitor.add_event('error', f'[{account_name}] Unable to move spam email: {exc}', {'account': account_name})

    def _fetch_google_unread_messages(self, service: Any, query: str, download_max: int) -> List[Dict[str, Any]]:
        emails: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        remaining = max(1, download_max)
        while remaining > 0:
            params = {
                'userId': 'me',
                'q': query,
                'maxResults': min(remaining, 100),
            }
            if page_token:
                params['pageToken'] = page_token
            response = service.users().messages().list(**params).execute()
            message_refs = response.get('messages') or []
            if not message_refs:
                break
            for ref in message_refs:
                msg_id = ref.get('id')
                if not msg_id:
                    continue
                detail = service.users().messages().get(userId='me', id=msg_id, format='raw').execute()
                raw_payload = detail.get('raw')
                if not raw_payload:
                    continue
                raw_bytes = base64.urlsafe_b64decode(raw_payload.encode('utf-8'))
                msg = email.message_from_bytes(raw_bytes)
                body = find_body_in_email(msg)
                emails.append(
                    {
                        'gmail_id': msg_id,
                        'id': msg.get('Message-ID') or msg.get('message-id') or msg_id,
                        'from': msg.get('from'),
                        'to': msg.get('to'),
                        'date': msg.get('date'),
                        'subject': self._decode_subject(msg.get('subject')),
                        'body_plain_text': body['plain_text_body'],
                        'body_html': body['html_body'],
                    }
                )
                if len(emails) >= download_max:
                    return emails
            page_token = response.get('nextPageToken')
            if not page_token:
                break
            remaining = download_max - len(emails)
        return emails

    def _build_gmail_service(self, credentials_path: str, token_path: str):
        if not all((Credentials, InstalledAppFlow, GoogleRequest, google_build)):
            raise RuntimeError('Google client libraries are missing. Install google-api-python-client and google-auth-oauthlib.')
        credentials_path = os.path.expanduser(credentials_path) if credentials_path else credentials_path
        token_path = os.path.expanduser(token_path) if token_path else token_path
        creds = None
        if token_path and os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, GOOGLE_SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(GoogleRequest())
            else:
                if not credentials_path or not os.path.exists(credentials_path):
                    raise RuntimeError('Google OAuth credentials file not found')
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, GOOGLE_SCOPES)
                auth_mode = os.environ.get('SPAMBUTCHER_GOOGLE_AUTH_MODE', '').lower()
                if auth_mode == 'console':
                    creds = flow.run_console()
                else:
                    creds = flow.run_local_server(port=0)
            if token_path:
                token_dir = os.path.dirname(token_path)
                if token_dir:
                    os.makedirs(token_dir, exist_ok=True)
                with open(token_path, 'w', encoding='utf-8') as token_file:
                    token_file.write(creds.to_json())
        return google_build('gmail', 'v1', credentials=creds, cache_discovery=False)

    def _normalize_gmail_label_name(self, label_name: str) -> str:
        if not label_name:
            return ''
        normalized = label_name.strip()
        mapping = {
            '[gmail]/spam': 'SPAM',
            '[gmail]/trash': 'TRASH',
            '[gmail]/inbox': 'INBOX',
            'spam': 'SPAM',
            'trash': 'TRASH',
            'inbox': 'INBOX',
        }
        return mapping.get(normalized.lower(), normalized)

    def _ensure_gmail_label_id(self, service: Any, label_name: str, account_name: str) -> Optional[str]:
        normalized = self._normalize_gmail_label_name(label_name)
        if not normalized:
            return None
        cache = self._gmail_label_cache.setdefault(account_name, {})
        cache_key = normalized.lower()
        if cache_key in cache:
            return cache[cache_key]
        labels = service.users().labels().list(userId='me').execute().get('labels', [])
        for label in labels:
            name = (label.get('name') or '').lower()
            if name == cache_key:
                cache[cache_key] = label.get('id')
                return cache[cache_key]
        body = {'name': normalized}
        created = service.users().labels().create(userId='me', body=body).execute()
        label_id = created.get('id')
        cache[cache_key] = label_id
        return label_id

    def _move_google_message_to_label(self, service: Any, message_id: str, label_name: str, account_name: str) -> None:
        label_id = self._ensure_gmail_label_id(service, label_name, account_name)
        if not label_id:
            raise RuntimeError('Unable to resolve Gmail label')
        body = {
            'addLabelIds': [label_id],
            'removeLabelIds': ['INBOX', 'UNREAD'],
        }
        service.users().messages().modify(userId='me', id=message_id, body=body).execute()
        self.monitor.add_event('info', f'[{account_name}] Moved email to spam folder', {'folder': label_name, 'account': account_name})


class ProcessorWorker:
    def __init__(
        self,
        processor: EmailProcessor,
        monitor: ProcessingMonitor,
        config_manager: ConfigManager,
    ) -> None:
        self.processor = processor
        self.monitor = monitor
        self.config_manager = config_manager
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.is_running():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.monitor.mark_run_start()
            processed = 0
            try:
                processed = self.processor.process_new_messages()
            except Exception as exc:
                self.monitor.mark_error(f'Processing crashed: {exc}')
            finally:
                self.monitor.mark_run_end(processed)
            interval = int(self.config_manager.get_config().get('poll_interval_seconds', 600))
            interval = max(30, interval)
            self._wait(interval)

    def _wait(self, seconds: int) -> None:
        waited = 0
        while waited < seconds and not self._stop_event.is_set():
            time.sleep(1)
            waited += 1

    def run_once(self) -> None:
        threading.Thread(target=self._run_once_task, daemon=True).start()

    def _run_once_task(self) -> None:
        self.monitor.mark_run_start()
        processed = 0
        try:
            processed = self.processor.process_new_messages()
        except Exception as exc:
            self.monitor.mark_error(f'Processing crashed: {exc}')
        finally:
            self.monitor.mark_run_end(processed)

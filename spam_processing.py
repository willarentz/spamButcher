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

import requests
from bs4 import BeautifulSoup


DATABASE_PATH = 'emails.db'
CONFIG_PATH = 'config.json'


DEFAULT_CONFIG: Dict[str, Any] = {
    "imap_host": "",
    "imap_username": "",
    "imap_password": "",
    "imap_folder": "INBOX",
    "spam_folder": "SpamAI",
    "download_max": 50,
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

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
        else:
            data = {}
        config = deep_merge(json.loads(json.dumps(DEFAULT_CONFIG)), data)
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
        conn.execute(
            '''CREATE TABLE IF NOT EXISTS emails
               (id TEXT PRIMARY KEY, subject TEXT, date TEXT, from_email TEXT,
                to_email TEXT, html_body TEXT, plain_text_body TEXT, is_spam BOOLEAN)'''
        )
        conn.commit()
        conn.close()

    def email_exists(self, message_id: str) -> bool:
        conn = self._connect()
        cur = conn.execute('SELECT 1 FROM emails WHERE id=?', (message_id,))
        row = cur.fetchone()
        conn.close()
        return row is not None

    def insert_email(self, email_payload: Dict[str, Any]) -> None:
        conn = self._connect()
        conn.execute(
            'INSERT OR IGNORE INTO emails VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (
                email_payload['id'],
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
            'SELECT id, subject, date, from_email, to_email, is_spam FROM emails '
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

    def get_email(self, message_id: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        cur = conn.execute(
            'SELECT * FROM emails WHERE id=?',
            (message_id,),
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return dict(row)


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

    def process_new_messages(self) -> int:
        config = self.config_manager.get_config()
        required = [
            field
            for field in ('imap_host', 'imap_username', 'imap_password')
            if not config.get(field)
        ]
        if required:
            self.monitor.add_event(
                'warning',
                f"IMAP settings incomplete: {', '.join(required)}",
            )
            return 0

        download_max = int(config.get('download_max', 50) or 50)
        mail = None
        processed = 0
        try:
            mail = imaplib.IMAP4_SSL(config['imap_host'])
            mail.login(config['imap_username'], config['imap_password'])
            folder = config.get('imap_folder', 'INBOX')
            status, _ = mail.select(folder)
            if status != 'OK':
                raise RuntimeError(f'Unable to select folder {folder}')
            unread = self._fetch_unread_messages(mail, download_max)
            if not unread:
                self.monitor.add_event('info', 'No unread emails found')
                return 0
            client = build_llm_client(config)
            for message in unread:
                if self.database.email_exists(message['id']):
                    continue
                prompt = formatting_function(message)
                try:
                    raw = client.generate(prompt, config.get('system_prompt', ''))
                    classification = self._interpret_response(raw)
                except Exception as exc:
                    self.monitor.add_event(
                        'error',
                        f"Failed to classify email {message['subject']}: {exc}",
                    )
                    continue
                message['isSpam'] = classification['is_spam']
                self.database.insert_email(message)
                processed += 1
                level = 'warning' if classification['is_spam'] else 'info'
                self.monitor.add_event(
                    level,
                    f"Email '{message['subject']}' flagged as "
                    f"{'spam' if classification['is_spam'] else 'ham'}",
                    {'reason': classification.get('reason', '')},
                )
                if classification['is_spam']:
                    self._move_to_spam(mail, message['e_id'], config.get('spam_folder', 'SpamAI'))
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

    def _move_to_spam(self, mail: imaplib.IMAP4_SSL, e_id: bytes, folder: str) -> None:
        try:
            status, _ = mail.copy(e_id, folder)
            if status != 'OK':
                raise RuntimeError('copy failed')
            mail.store(e_id, '+FLAGS', '\\Deleted')
            self.monitor.add_event('info', 'Moved email to spam folder', {'folder': folder})
        except Exception as exc:
            self.monitor.add_event('error', f'Unable to move spam email: {exc}')


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

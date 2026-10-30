"""Pip-Beast web UI on http://localhost:8000 (Python standard library only).

    ./run.sh ui
    python web_ui.py --host 0.0.0.0 --port 8000     # reachable from the LAN
    python web_ui.py --core                         # start with satellites off

Routes:
    GET  /                  the UI (web/index.html)
    GET  /api/status        model, device, satellite state
    POST /api/chat          {message, temperature, top_k, max_new_tokens} -> text/event-stream
                            events: token {text}, done {satellites}, error {message}
    POST /api/satellites    {enabled: bool}
    POST /api/reset         clear conversation history and satellite hormone state
"""
import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from generate import load_checkpoint, stream_complete
from tokenizer import BeastTokenizer

HERE = Path(__file__).resolve().parent
INDEX_PATH = HERE / 'web' / 'index.html'


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


class Session:
    """One loaded model shared by all requests. Generation is serialized with a lock."""

    def __init__(self, a):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, _ = load_checkpoint(a.ckpt, self.device)
        self.tok = BeastTokenizer(a.tokenizer)
        self.params = self.model.num_params()
        self.context = self.model.cfg.max_seq_len
        self.lock = threading.Lock()
        self.history = ''
        self.orch = None
        self.sat_error = None
        self.use_sat = False
        try:
            from satellites.orchestrator import Orchestrator
            self.orch = Orchestrator(self.model, self.tok, memory_path=a.memory,
                                     device=self.device, max_new_tokens=a.max_new_tokens)
            self.use_sat = not a.core
        except Exception as e:  # satellites are optional, same as chat.py
            self.sat_error = str(e)


class Handler(BaseHTTPRequestHandler):
    session = None  # set in main()
    server_version = 'PipBeast/1.0'

    def log_message(self, fmt, *args):
        sys.stderr.write('%s - %s\n' % (self.address_string(), fmt % args))

    def _send_json(self, code, obj):
        data = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        n = int(self.headers.get('Content-Length') or 0)
        if n > 1_000_000:
            raise ValueError('body too large')
        raw = self.rfile.read(n) if n else b''
        return json.loads(raw or b'{}')

    def do_GET(self):
        s = self.session
        if self.path in ('/', '/index.html'):
            data = INDEX_PATH.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)
        elif self.path == '/api/status':
            self._send_json(200, {
                'model': 'Pip-Beast',
                'params': s.params,
                'device': s.device,
                'context': s.context,
                'satellites_available': s.orch is not None,
                'satellites_on': s.use_sat and s.orch is not None,
                'satellite_error': s.sat_error,
            })
        else:
            self._send_json(404, {'error': 'not found'})

    def do_POST(self):
        try:
            body = self._read_json()
        except (ValueError, json.JSONDecodeError):
            return self._send_json(400, {'error': 'invalid JSON body'})
        if self.path == '/api/chat':
            return self._chat(body)
        if self.path == '/api/satellites':
            return self._satellites(body)
        if self.path == '/api/reset':
            return self._reset()
        return self._send_json(404, {'error': 'not found'})

    def _satellites(self, body):
        s = self.session
        if s.orch is None:
            return self._send_json(409, {'error': 'satellites unavailable', 'detail': s.sat_error})
        s.use_sat = bool(body.get('enabled', False))
        return self._send_json(200, {'satellites_on': s.use_sat})

    def _reset(self):
        s = self.session
        with s.lock:
            s.history = ''
            if s.orch is not None:
                try:
                    from satellites.endocrine import BASELINE
                    s.orch.endo.levels = dict(BASELINE)
                except Exception:
                    pass
        return self._send_json(200, {'ok': True})

    def _chat(self, body):
        s = self.session
        message = str(body.get('message', '')).strip()[:4000]
        if not message:
            return self._send_json(400, {'error': 'empty message'})
        try:
            temperature = clamp(float(body.get('temperature', 0.8)), 0.05, 2.0)
            top_k = clamp(int(body.get('top_k', 40)), 0, 200)
            max_new = clamp(int(body.get('max_new_tokens', 200)), 1, 1024)
        except (TypeError, ValueError):
            return self._send_json(400, {'error': 'invalid sampling parameters'})

        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Accel-Buffering', 'no')
        self.end_headers()

        def emit(event, payload):
            frame = 'event: %s\ndata: %s\n\n' % (event, json.dumps(payload))
            self.wfile.write(frame.encode('utf-8'))
            self.wfile.flush()

        with s.lock:
            try:
                satellites_used = s.use_sat and s.orch is not None
                if satellites_used:
                    reply = s.orch.respond(message).strip()
                    for word in reply.split(' '):
                        emit('token', {'text': word + ' '})
                else:
                    prompt = s.history + f'User: {message}\nPip:'
                    parts = []
                    for piece in stream_complete(s.model, s.tok, prompt, max_new, temperature,
                                                 top_k, s.device):
                        parts.append(piece)
                        emit('token', {'text': piece})
                    reply = ''.join(parts).strip()
                    s.history = prompt + ' ' + reply + '\n'
                emit('done', {'satellites': satellites_used})
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as e:
                try:
                    emit('error', {'message': str(e)})
                except OSError:
                    pass


def main():
    p = argparse.ArgumentParser(description='Pip-Beast web UI')
    p.add_argument('--ckpt', default='checkpoints/best.pt')
    p.add_argument('--tokenizer', default='data/tokenizer.json')
    p.add_argument('--memory', default='checkpoints/episodes.json')
    p.add_argument('--host', default='127.0.0.1', help='use 0.0.0.0 to expose on the LAN')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--core', action='store_true', help='start with satellites off')
    p.add_argument('--max_new_tokens', type=int, default=200)
    a = p.parse_args()

    if not Path(a.ckpt).exists():
        sys.exit(f'No checkpoint at {a.ckpt}. Train first: ./run.sh train')
    if not INDEX_PATH.exists():
        sys.exit(f'missing UI file: {INDEX_PATH}')

    Handler.session = Session(a)
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    server.daemon_threads = True
    print(f'Pip-Beast UI: http://localhost:{a.port} (bound to {a.host}) | device {Handler.session.device}')
    print(f'satellites: {"on" if Handler.session.use_sat else "off"}'
          + (f' (unavailable: {Handler.session.sat_error})' if Handler.session.sat_error else ''))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

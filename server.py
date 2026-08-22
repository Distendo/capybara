#!/usr/bin/env python3
"""Capybara gateway - web UI, OpenAI-compatible proxy and engine supervisor.

The gateway is a single long-lived process that owns one llama.cpp
`llama-server` child process bound to an internal loopback port. It exposes:

* ``GET /``            - the built-in chat UI (ui/index.html)
* ``GET /api/status``  - engine/model status as JSON
* ``GET /api/models``  - installed models as JSON
* ``POST /api/use``    - hot-swap the loaded model (JSON ``{"model": name}``)
* everything else      - transparently proxied to the engine, including
  streaming Server-Sent Events responses

Because the public endpoint lives in the gateway, clients never observe
engine restarts: swapping models keeps the API up while the child restarts.
Only the Python standard library is used.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capybara as cb  # noqa: E402

HOP_HEADERS = {"content-length", "transfer-encoding", "connection", "keep-alive",
               "host", "accept-encoding"}


class EngineManager:
    """Spawn, monitor and hot-swap the llama-server child process."""

    def __init__(self, settings: cb.Settings) -> None:
        self.settings = settings
        self.proc: Optional[subprocess.Popen] = None
        self.model: Optional[Path] = None
        self.started_at: float = 0.0
        self.swapping = False

    @property
    def internal_url(self) -> str:
        return f"http://127.0.0.1:{self.settings.engine_port}"

    def _command(self, model: Path) -> list:
        meta = cb.sidecar_for(model) or {}
        context = int((meta.get("params") or {}).get("num_ctx", self.settings.context))
        return [
            str(self.settings.server_bin),
            "--model", str(model),
            "--host", "127.0.0.1",
            "--port", str(self.settings.engine_port),
            "--threads", str(self.settings.threads),
            "--ctx-size", str(context),
            "--batch-size", str(self.settings.batch),
            "--ubatch-size", str(self.settings.ubatch),
            "--n-gpu-layers", str(self.settings.gpu_layers),
            "--parallel", "1",
            "--cont-batching",
            "--flash-attn", "on",
        ]

    def _spawn(self, model: Path) -> None:
        """Start the engine child and wait until it answers /health."""
        self.settings.run_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.settings.log_file, "ab")
        try:
            self.proc = subprocess.Popen(self._command(model), stdout=log, stderr=log)
        except OSError as exc:
            log.close()
            raise RuntimeError(f"failed to launch engine: {exc}") from exc
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.proc.poll() is not None:
                break
            if self._healthy():
                self.model = model
                self.started_at = time.time()
                self._write_state()
                return
            time.sleep(0.25)
        tail = ""
        if self.settings.log_file.exists():
            lines = self.settings.log_file.read_text(errors="replace").splitlines()
            tail = "\n".join(lines[-8:])
        self.shutdown()
        raise RuntimeError(f"engine failed to start; see {self.settings.log_file}\n{tail}")

    def _healthy(self, timeout: float = 1.0) -> bool:
        try:
            req = urllib.request.Request(f"{self.internal_url}/health")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            return False

    def _write_state(self) -> None:
        state = {
            "mode": "gateway",
            "gateway_pid": os_pid(),
            "pid": self.proc.pid if self.proc else None,
            "model": self.model.name if self.model else None,
            "path": str(self.model) if self.model else None,
            "host": self.settings.host,
            "port": self.settings.port,
            "engine_port": self.settings.engine_port,
            "started_at": self.started_at,
        }
        try:
            self.settings.run_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.settings.state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(state))
            tmp.replace(self.settings.state_file)
        except OSError:
            pass

    def start(self, model: Path) -> None:
        if self.proc is not None:
            raise RuntimeError("engine already running")
        self._spawn(model)

    def swap(self, model: Path) -> None:
        """Replace the loaded model without dropping the public endpoint."""
        if self.proc is not None and self.model is not None \
                and self.model.resolve() == model.resolve() and self._healthy():
            return
        self.swapping = True
        try:
            self.stop()
            self._spawn(model)
        finally:
            self.swapping = False

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            proc.terminate()
        except OSError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
            proc.wait(timeout=5)
        self.model = None


def os_pid() -> int:
    """Current process id."""
    return os.getpid()


class Gateway(BaseHTTPRequestHandler):
    """HTTP request handler wired to an EngineManager instance."""

    manager: EngineManager
    ui_html: bytes

    protocol_version = "HTTP/1.1"
    server_version = "Capybara"

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence per-request access logs (engine log has the details)."""
        return

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_bytes(200, self.ui_html, "text/html; charset=utf-8")
        elif path == "/api/status":
            self._send_json(200, self.status())
        elif path == "/api/models":
            self._send_json(200, self.models())
        elif path == "/favicon.ico":
            svg = (b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                   b'<text y="80" font-size="80">\U0001F42D</text></svg>')
            self._send_bytes(200, svg, "image/svg+xml")
        else:
            self.proxy()

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/use":
            self.api_use()
        else:
            self.proxy()

    # ------------------------------------------------------------------ utils
    def send_error_reply(self, code: int, message: str) -> None:
        self._send_json(code, {"error": message})

    def _send_bytes(self, code: int, payload: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, code: int, obj: Any) -> None:
        payload = json.dumps(obj).encode()
        self._send_bytes(code, payload, "application/json")

    def read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length > 0 else b""

    # --------------------------------------------------------------- endpoints
    def status(self) -> Dict[str, Any]:
        mgr = self.manager
        ready = mgr.proc is not None and mgr._healthy()
        return {
            "status": "ok" if ready else ("loading" if mgr.swapping else "down"),
            "model": mgr.model.name if mgr.model else None,
            "swapping": mgr.swapping,
            "uptime": int(time.time() - mgr.started_at) if mgr.started_at else 0,
            "port": mgr.settings.port,
            "engine_port": mgr.settings.engine_port,
            "models_dir": str(mgr.settings.models),
            "version": cb.VERSION,
        }

    def models(self) -> Dict[str, Any]:
        settings = self.manager.settings
        loaded_key = None
        if self.manager.model is not None:
            match = cb.SHARD_RE.search(self.manager.model.name)
            stem = self.manager.model.stem
            loaded_key = match.group(1) if match else stem
        entries = []
        for name, size in cb.group_models(cb.list_models(settings)):
            entries.append({"name": name, "size": size, "loaded": name == loaded_key})
        return {"models": entries}

    def api_use(self) -> None:
        """Switch the loaded model: POST {"model": "<installed name>"}."""
        settings = self.manager.settings
        try:
            wanted = json.loads(self.read_body().decode("utf-8")).get("model", "")
        except ValueError:
            self.send_error_reply(400, "invalid JSON body")
            return
        model = cb.resolve_model_or_alias(settings, str(wanted))
        if model is None:
            installed = ", ".join(name for name, _ in cb.group_models(
                cb.list_models(settings))) or "(none)"
            self.send_error_reply(404, f"model not installed: {wanted}. Installed: {installed}")
            return
        try:
            self.manager.swap(model)
        except RuntimeError as exc:
            self.send_error_reply(503, str(exc))
            return
        self._send_json(200, {"ok": True, "model": model.name})

    # ------------------------------------------------------------------ proxy
    def proxy(self) -> None:
        """Forward the request to the engine, streaming responses through."""
        mgr = self.manager
        body = self.read_body()
        headers = {}
        for key, value in self.headers.items():
            if key.lower() not in HOP_HEADERS:
                headers[key] = value
        if self.command == "POST" and self.path.split("?", 1)[0].endswith("/chat/completions"):
            try:
                payload: Dict[str, Any] = json.loads(body.decode("utf-8"))
                body = json.dumps(inject_sidecar(mgr, payload)).encode("utf-8")
            except ValueError:
                pass  # let the engine reject malformed bodies
        url = f"{mgr.internal_url}{self.path}"
        req = urllib.request.Request(url, data=body if body else None,
                                     headers=headers, method=self.command)
        try:
            resp = urllib.request.urlopen(req, timeout=None)
        except urllib.error.HTTPError as exc:
            self._send_bytes(exc.code, exc.read(), exc.headers.get_content_type()
                             or "application/json")
            return
        except (urllib.error.URLError, OSError):
            hint = "model is loading - retry shortly" if mgr.swapping else "engine unavailable"
            self.send_error_reply(503, hint)
            return

        ctype = resp.headers.get("Content-Type", "application/octet-stream")
        streaming = "text/event-stream" in ctype
        self.send_response(resp.status)
        self.send_header("Content-Type", ctype)
        if streaming:
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                while True:
                    chunk = resp.read(1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                resp.close()
                self.close_connection = True
        else:
            payload = resp.read()
            resp.close()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


def inject_sidecar(mgr: EngineManager, body: Dict[str, Any]) -> Dict[str, Any]:
    """Apply 'capybara create' defaults (system prompt, params) to a request."""
    if mgr.model is None or "messages" not in body:
        return body
    system_msgs, params = cb.request_context(mgr.model)
    messages = body.get("messages") or []
    has_system = any(m.get("role") == "system" for m in messages)
    if system_msgs and not has_system:
        body["messages"] = system_msgs + messages
    for key, value in params.items():
        body.setdefault(key, value)
    return body


def main(argv: Optional[list] = None) -> int:
    """Gateway entry point: start engine, then serve forever."""
    args = list(sys.argv[1:] if argv is None else argv)
    wanted = args[args.index("--model") + 1] if "--model" in args else None
    settings = cb.load_settings()
    mgr = EngineManager(settings)

    model = None
    if wanted:
        model = cb.resolve_model(settings, wanted)
    if model is None:
        installed = cb.list_models(settings)
        if not installed:
            print(f"capybara: no models installed - try: capybara pull smollm",
                  file=sys.stderr)
            return 1
        model = installed[0]

    ui_candidates = [Path(__file__).resolve().parent / "ui" / "index.html",
                     settings.home / "ui" / "index.html"]
    ui_path = next((p for p in ui_candidates if p.exists()), None)
    Gateway.ui_html = ui_path.read_bytes() if ui_path else (
        b"<h1>Capybara</h1><p>ui/index.html not found</p>")
    Gateway.manager = mgr

    try:
        mgr.start(model)
    except RuntimeError as exc:
        print(f"capybara: {exc}", file=sys.stderr)
        return 1

    try:
        server = ThreadingHTTPServer((settings.host, settings.port), Gateway)
    except OSError as exc:
        mgr.stop()
        settings.state_file.unlink(missing_ok=True)
        print(f"capybara: cannot bind {settings.host}:{settings.port}: {exc} - "
              f"is another Capybara or Ollama running?", file=sys.stderr)
        return 1
    server.daemon_threads = True

    def shutdown(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    print(f"Capybara {cb.VERSION} serving {mgr.model.name} "
          f"on http://{settings.host}:{settings.port} (UI: /, API: /v1)")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        mgr.stop()
        settings.state_file.unlink(missing_ok=True)
        (settings.run_dir / "server.pid").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

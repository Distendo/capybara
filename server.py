#!/usr/bin/env python3
"""Capybara gateway - Ollama-compatible API, OpenAI proxy, engine supervisor.

The gateway is a single long-lived process that owns one llama.cpp
`llama-server` child process bound to an internal loopback port. It exposes:

 * ``GET  /``               - service descriptor (JSON)
 * ``GET  /api/version``    - version banner
 * ``GET  /api/status``     - engine/model status as JSON
 * ``GET  /api/models``     - installed models (native shape)
 * ``GET  /api/tags``       - installed models (Ollama shape)
 * ``GET  /api/ps``         - running/loaded models (Ollama shape)
 * ``POST /api/use``        - hot-swap the loaded model
 * ``POST /api/chat``       - Ollama-native chat (streaming NDJSON)
 * ``POST /api/generate``   - Ollama-native completion (streaming NDJSON)
 * ``POST /api/show``       - model details (Ollama shape)
 * ``POST /api/pull``       - pull a model (streaming progress NDJSON)
 * ``POST /v1/*``           - OpenAI-compatible passthrough to the engine
 * ``DELETE /api/delete``   - remove an installed model
 * everything else          - transparently proxied to the engine, including
   streaming Server-Sent Events responses

Requests naming a different model hot-swap the engine automatically, and an
idle keep-alive timer unloads the model to free memory (configurable via
CAPYBARA_KEEP_ALIVE, config ``runtime.keep_alive``, or per-request
``keep_alive``). Only the Python standard library is used.
"""
from __future__ import annotations

import json
import math
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capybara as cb  # noqa: E402

HOP_HEADERS = {"content-length", "transfer-encoding", "connection", "keep-alive",
               "host", "accept-encoding"}


def iso_now(offset_seconds: float = 0.0) -> str:
    """UTC ISO-8601 timestamp like Ollama's created_at fields."""
    when = time.gmtime(time.time() + offset_seconds)
    return time.strftime("%Y-%m-%dT%H:%M:%S", when) + ".%03dZ" % (
        int((time.time() + offset_seconds) % 1 * 1000),)


def file_digest(path: Path) -> str:
    """Cheap stable pseudo-digest (Ollama clients treat it as opaque)."""
    st = path.stat()
    return f"sha256:{int(st.st_size):016x}{int(st.st_mtime):012x}"


def param_size_guess(name: str) -> str:
    """Extract a parameter-size label like '7B' or '135M' from a model name."""
    match = cb.SHARD_RE.search(name)
    stem = (match.group(1) if match else Path(name).stem).lower()
    hit = next((m for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([bm])(?![a-z0-9])",
                                       stem)), None)
    return f"{hit.group(1)}{hit.group(2).upper()}" if hit else ""


def quant_guess(name: str) -> str:
    """Extract a quantisation label like 'Q4_K_M' from a model name."""
    stem = Path(name).stem.upper()
    hit = cb.QUANT_SUFFIX_RE.search(stem)
    return hit.group(0).lstrip("-._") if hit else ""


class EngineManager:
    """Spawn, monitor, hot-swap and idle-unload the llama-server child."""

    def __init__(self, settings: cb.Settings) -> None:
        self.settings = settings
        self.proc: Optional[subprocess.Popen] = None
        self.model: Optional[Path] = None
        self.started_at: float = 0.0
        self.swapping = False
        self.unloaded_idle = False
        self.keep_alive = cb.parse_keep_alive(settings.keep_alive_raw, default=300.0)
        self.last_used = time.monotonic()
        self.lock = threading.RLock()
        threading.Thread(target=self._watchdog, daemon=True).start()

    # ------------------------------------------------------------ lifecycle
    @property
    def internal_url(self) -> str:
        return f"http://127.0.0.1:{self.settings.engine_port}"

    def touch(self, keep_alive: Optional[Any] = None) -> None:
        """Mark activity now, optionally adopting a new keep-alive value."""
        self.last_used = time.monotonic()
        if keep_alive is not None:
            self.keep_alive = cb.parse_keep_alive(keep_alive, default=self.keep_alive)

    def _watchdog(self) -> None:
        """Unload the engine after keep_alive seconds without requests."""
        while True:
            time.sleep(1.0)
            with self.lock:
                if (self.proc is None or self.swapping
                        or self.keep_alive == math.inf):
                    continue
                if time.monotonic() - self.last_used >= self.keep_alive:
                    self.unloaded_idle = True
                    self._stop_locked()

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
                self.unloaded_idle = False
                self.touch()
                self._write_state()
                return
            time.sleep(0.25)
        tail = ""
        if self.settings.log_file.exists():
            lines = self.settings.log_file.read_text(errors="replace").splitlines()
            tail = "\n".join(lines[-8:])
        self._stop_locked()
        raise RuntimeError(f"engine failed to start; see {self.settings.log_file}\n{tail}")

    def _healthy(self, timeout: float = 1.0) -> bool:
        try:
            req = urllib.request.Request(f"{self.internal_url}/health")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            return False

    def _write_state(self, extra: Optional[Dict[str, Any]] = None) -> None:
        state = {
            "mode": "gateway",
            "gateway_pid": os.getpid(),
            "pid": self.proc.pid if self.proc else None,
            "model": self.model.name if self.model else None,
            "path": str(self.model) if self.model else None,
            "host": self.settings.host,
            "port": self.settings.port,
            "engine_port": self.settings.engine_port,
            "started_at": self.started_at,
            "idle_unloaded": self.unloaded_idle,
        }
        if extra:
            state.update(extra)
        try:
            self.settings.run_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.settings.state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(state))
            tmp.replace(self.settings.state_file)
        except OSError:
            pass

    def ensure(self, model: Path) -> None:
        """Guarantee `model` is the one loaded, spawning/swapping as needed."""
        with self.lock:
            self.touch()
            if self.proc is not None and self.model is not None \
                    and self.model.resolve() == model.resolve() and self._healthy():
                return
            if self.proc is not None:
                self._stop_locked()
            self.swapping = True
            try:
                self._spawn(model)
            finally:
                self.swapping = False

    def start(self, model: Path) -> None:
        with self.lock:
            if self.proc is not None:
                raise RuntimeError("engine already running")
            self.swapping = True
            try:
                self._spawn(model)
            finally:
                self.swapping = False

    def swap(self, model: Path) -> None:
        """Replace the loaded model without dropping the public endpoint."""
        self.ensure(model)

    def _stop_locked(self) -> None:
        proc, self.proc = self.proc, None
        self.model = None
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
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self._write_state()

    def stop(self) -> None:
        with self.lock:
            self._stop_locked()

    def usage_snapshot(self) -> Dict[str, Any]:
        """Info for /api/ps and /api/status about the currently loaded model."""
        with self.lock:
            if self.model is None:
                return {"models": []}
            path = self.model
            remaining: Optional[float] = None
            if self.keep_alive != math.inf:
                remaining = max(0.0, self.keep_alive
                                - (time.monotonic() - self.last_used))
            entry: Dict[str, Any] = {
                "name": path.name,
                "model": path.name,
                "size": path.stat().st_size,
                "size_vram": path.stat().st_size if self.settings.gpu_layers > 0 else 0,
                "digest": file_digest(path),
                "details": {
                    "format": "gguf",
                    "parameter_size": param_size_guess(path.name),
                    "quantization_level": quant_guess(path.name),
                },
                "expires_at": iso_now(remaining) if remaining is not None else None,
            }
            return {"models": [entry]}


class Gateway(BaseHTTPRequestHandler):
    """HTTP request handler wired to an EngineManager instance."""

    manager: EngineManager

    protocol_version = "HTTP/1.1"
    server_version = "Capybara"

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence per-request access logs (engine log has the details)."""
        return

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_json(200, {
                "service": "capybara",
                "version": cb.VERSION,
                "api": "Ollama-compatible (/api/chat, /api/generate, /api/tags)"
                       " + OpenAI-compatible (/v1/chat/completions)",
                "ui": "run 'capybara ui' for the Open WebUI frontend",
            })
        elif path == "/api/version":
            self._send_json(200, {"version": cb.VERSION})
        elif path == "/api/status":
            self._send_json(200, self.status())
        elif path == "/api/models":
            self._send_json(200, self.models())
        elif path == "/api/tags":
            self._send_json(200, self.api_tags())
        elif path == "/api/ps":
            self._send_json(200, self.manager.usage_snapshot())
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
        elif path == "/api/chat":
            self.api_chat()
        elif path == "/api/generate":
            self.api_generate()
        elif path == "/api/show":
            self.api_show()
        elif path == "/api/pull":
            self.api_pull()
        else:
            self.proxy()

    def do_DELETE(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/api/delete", "/api/delete/"):
            self.api_delete()
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
        if ready:
            state = "ok"
        elif mgr.swapping:
            state = "loading"
        elif mgr.unloaded_idle:
            state = "idle"
        else:
            state = "down"
        return {
            "status": state,
            "model": mgr.model.name if mgr.model else None,
            "swapping": mgr.swapping,
            "idle_unloaded": mgr.unloaded_idle,
            "keep_alive": (None if mgr.keep_alive == math.inf else mgr.keep_alive),
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

    # ------------------------------------------------- Ollama-compatible API
    def api_tags(self) -> Dict[str, Any]:
        """Installed models in Ollama's /api/tags shape."""
        settings = self.manager.settings
        seen: Dict[str, Path] = {}
        total: Dict[str, int] = {}
        mtime: Dict[str, float] = {}
        for path in cb.list_models(settings):
            match = cb.SHARD_RE.search(path.name)
            key = match.group(1) if match else path.stem
            seen.setdefault(key, path)
            total[key] = total.get(key, 0) + path.stat().st_size
            mtime[key] = max(mtime.get(key, 0.0), path.stat().st_mtime)
        entries = []
        for key in sorted(total):
            rep = seen[key]
            entries.append({
                "name": f"{key}:latest",
                "model": f"{key}:latest",
                "modified_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(mtime[key])),
                "size": total[key],
                "digest": file_digest(rep),
                "details": {
                    "format": "gguf",
                    "family": "",
                    "families": [],
                    "parameter_size": param_size_guess(rep.name),
                    "quantization_level": quant_guess(rep.name),
                },
            })
        return {"models": entries}

    def _resolve_for_request(self, wanted: Any) -> Path:
        """Resolve a requested model and make sure it is loaded (auto-swap)."""
        mgr = self.manager
        name = str(wanted or "").strip()
        model = cb.resolve_model_or_alias(mgr.settings, name) if name else mgr.model
        if model is None:
            if name:
                raise ApiError(404, f"model '{name}' not found, try pulling it first")
            raise ApiError(400, "model is required - none loaded and no model named")
        try:
            mgr.ensure(model)
        except RuntimeError as exc:
            raise ApiError(503, str(exc))
        return model

    def _start_ndjson(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _ndjson(self, obj: Dict[str, Any]) -> None:
        self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        self.wfile.flush()

    def api_chat(self) -> None:
        """Ollama POST /api/chat - messages in, NDJSON chunks out."""
        try:
            body = json.loads(self.read_body().decode("utf-8") or "{}")
        except ValueError:
            body = None
        if not isinstance(body, dict):
            self.send_error_reply(400, "invalid JSON body")
            return
        raw_in = body.get("messages")
        if not isinstance(raw_in, list) or not raw_in:
            self.send_error_reply(400, "messages is required")
            return
        msgs = [{"role": str(m.get("role") or "user"),
                 "content": str(m.get("content") or "")}
                for m in raw_in if isinstance(m, dict)]
        try:
            model = self._resolve_for_request(body.get("model"))
        except ApiError as exc:
            self.send_error_reply(exc.code, exc.message)
            return
        self.manager.touch(keep_alive=body.get("keep_alive"))
        started = time.monotonic()
        sidecar_msgs, sidecar_params = cb.request_context(model)
        has_system = any(m["role"] == "system" for m in msgs)
        if sidecar_msgs and not has_system:
            msgs = sidecar_msgs + msgs
        payload: Dict[str, Any] = {"model": model.name, "messages": msgs,
                                   **sidecar_params, "stream": True,
                                   "stream_options": {"include_usage": True}}
        payload.update(cb.ollama_options_to_openai(body.get("options")))
        stream = bool(body.get("stream", True))
        if stream:
            self._start_ndjson()

        deltas: List[str] = []
        thoughts: List[str] = []
        usage: Optional[Dict[str, Any]] = None
        first_at: Optional[float] = None
        error: Optional[str] = None
        try:
            for obj in iter_sse(engine_stream(
                    self.manager, "/v1/chat/completions", payload)):
                self.manager.touch()
                if obj.get("usage"):
                    usage = obj["usage"]
                    continue
                try:
                    chunk = obj["choices"][0]["delta"]
                except (KeyError, IndexError, TypeError):
                    continue
                text = chunk.get("content") or ""
                thought = chunk.get("reasoning_content") or ""
                if not text and not thought:
                    continue
                if stream:
                    message: Dict[str, Any] = {"role": "assistant",
                                               "content": text}
                    if thought:
                        message["thinking"] = thought
                        thoughts.append(thought)
                    else:
                        deltas.append(text)
                    self._ndjson({"model": model.name, "created_at": iso_now(),
                                  "message": message, "done": False})
                elif thought:
                    thoughts.append(thought)
                else:
                    deltas.append(text)
                if first_at is None:
                    first_at = time.monotonic()
        except ApiError as exc:
            if not stream:
                error = exc.message
            else:
                try:
                    self._ndjson({"error": exc.message})
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                return

        ended = time.monotonic()
        eval_count = int(usage.get("completion_tokens",
                                   len(deltas) + len(thoughts))) if usage \
            else len(deltas) + len(thoughts)
        message = {"role": "assistant", "content": "" if error is None else error}
        if thoughts:
            message["thinking"] = "".join(thoughts)
        final: Dict[str, Any] = {
            "model": model.name,
            "created_at": iso_now(),
            "message": message,
            "done_reason": "stop" if error is None else "error",
            "done": True,
            "total_duration": int((ended - started) * 1e9),
            "eval_duration": int(((ended - first_at) if first_at else 0) * 1e9),
            "eval_count": eval_count,
        }
        if usage and usage.get("prompt_tokens"):
            final["prompt_eval_count"] = int(usage["prompt_tokens"])
        if stream and error is None:
            try:
                self._ndjson(final)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif error is None:
            final["message"]["content"] = "".join(deltas)
            self._send_json(200, final)
        elif not stream:
            self.send_error_reply(503, error)
        else:
            try:
                self._ndjson({"error": error})
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    def api_generate(self) -> None:
        """Ollama POST /api/generate - prompt in, NDJSON chunks out.

        With ``raw: true`` the prompt bypasses chat templating and hits the
        engine's completion endpoint directly.
        """
        try:
            body = json.loads(self.read_body().decode("utf-8") or "{}")
        except ValueError:
            body = None
        if not isinstance(body, dict):
            self.send_error_reply(400, "invalid JSON body")
            return
        prompt = str(body.get("prompt") or "")
        if not prompt:
            self.send_error_reply(400, "prompt is required")
            return
        try:
            model = self._resolve_for_request(body.get("model"))
        except ApiError as exc:
            self.send_error_reply(exc.code, exc.message)
            return
        self.manager.touch(keep_alive=body.get("keep_alive"))
        started = time.monotonic()
        raw_mode = bool(body.get("raw"))
        endpoint = "/v1/completions" if raw_mode else "/v1/chat/completions"
        payload: Dict[str, Any]
        if raw_mode:
            payload = {"model": model.name, "prompt": prompt,
                       "stream": True,
                       "stream_options": {"include_usage": True}}
            payload.update(cb.ollama_options_to_openai(body.get("options")))
        else:
            sidecar_msgs, sidecar_params = cb.request_context(model)
            system = body.get("system")
            msgs: List[Dict[str, str]] = []
            if system:
                msgs.append({"role": "system", "content": str(system)})
            elif sidecar_msgs:
                msgs.extend(sidecar_msgs)
            msgs.append({"role": "user", "content": prompt})
            payload = {"model": model.name, "messages": msgs, **sidecar_params,
                       "stream": True,
                       "stream_options": {"include_usage": True}}
            payload.update(cb.ollama_options_to_openai(body.get("options")))
        stream = bool(body.get("stream", True))

        deltas: List[str] = []
        thoughts: List[str] = []
        usage: Optional[Dict[str, Any]] = None
        first_at: Optional[float] = None
        error: Optional[str] = None
        field = "response"
        if stream:
            self._start_ndjson()
        try:
            for obj in iter_sse(engine_stream(self.manager, endpoint, payload)):
                self.manager.touch()
                if obj.get("usage"):
                    usage = obj["usage"]
                    continue
                try:
                    choice = obj["choices"][0]
                    delta = ((choice.get("delta") or {}).get("content")
                             or choice.get("text") or "")
                    thought = (choice.get("delta") or {}).get("reasoning_content") or ""
                except (KeyError, IndexError, TypeError):
                    continue
                if not delta and not thought:
                    continue
                if stream:
                    item: Dict[str, Any] = {"model": model.name,
                                            "created_at": iso_now(),
                                            field: delta, "done": False}
                    if thought:
                        item["thinking"] = thought
                        thoughts.append(thought)
                    else:
                        deltas.append(delta)
                    self._ndjson(item)
                elif thought:
                    thoughts.append(thought)
                else:
                    deltas.append(delta)
                if first_at is None:
                    first_at = time.monotonic()
        except ApiError as exc:
            if not stream:
                error = exc.message
            else:
                try:
                    self._ndjson({"error": exc.message})
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                return

        ended = time.monotonic()
        eval_count = int(usage.get("completion_tokens",
                                   len(deltas) + len(thoughts))) if usage \
            else len(deltas) + len(thoughts)
        final: Dict[str, Any] = {
            "model": model.name,
            "created_at": iso_now(),
            "done_reason": "stop" if error is None else "error",
            "done": True,
            "total_duration": int((ended - started) * 1e9),
            "eval_duration": int(((ended - first_at) if first_at else 0) * 1e9),
            "eval_count": eval_count,
        }
        final[field] = ""
        if thoughts:
            final["thinking"] = "".join(thoughts)
        if usage and usage.get("prompt_tokens"):
            final["prompt_eval_count"] = int(usage["prompt_tokens"])
        if stream and error is None:
            try:
                self._ndjson(final)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif error is None:
            final[field] = "".join(deltas)
            self._send_json(200, final)
        elif not stream:
            self.send_error_reply(503, error)
        else:
            try:
                self._ndjson({"error": error})
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    def api_show(self) -> None:
        """Ollama POST /api/show - best-effort model details."""
        try:
            body = json.loads(self.read_body().decode("utf-8") or "{}")
        except ValueError:
            body = None
        if not isinstance(body, dict):
            self.send_error_reply(400, "invalid JSON body")
            return
        settings = self.manager.settings
        model = cb.resolve_model_or_alias(settings, str(body.get("model") or ""))
        if model is None:
            self.send_error_reply(404, f"model '{body.get('model')}' not found")
            return
        meta = cb.sidecar_for(model) or {}
        params = meta.get("params") or {}
        parameters = "\n".join(f"{k:<20}{v}" for k, v in sorted(params.items()))
        self._send_json(200, {
            "license": "",
            "modelfile": meta.get("template") or "",
            "parameters": parameters,
            "template": meta.get("template") or "",
            "details": {
                "parent_model": meta.get("base") or "",
                "format": "gguf",
                "family": "",
                "families": [],
                "parameter_size": param_size_guess(model.name),
                "quantization_level": quant_guess(model.name),
            },
            "model_info": {
                "general.architecture": "llama",
                "general.file_type": quant_guess(model.name),
                "capybara.path": str(model),
                "capybara.size": model.stat().st_size,
                "capybara.system": meta.get("system") or "",
            },
        })

    def api_pull(self) -> None:
        """Ollama POST /api/pull - download progress as NDJSON lines."""
        try:
            body = json.loads(self.read_body().decode("utf-8") or "{}")
        except ValueError:
            body = None
        if not isinstance(body, dict):
            self.send_error_reply(400, "invalid JSON body")
            return
        spec = str(body.get("name") or body.get("model") or "").strip()
        if not spec:
            self.send_error_reply(400, "name is required")
            return
        settings = self.manager.settings
        stream = bool(body.get("stream", True))
        progress: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()

        def worker() -> None:
            try:
                cb.pull_model(settings, spec)
                progress.put({"status": "success"})
            except SystemExit as exc:  # die() inside pull_model
                progress.put({"status": "error", "error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                progress.put({"status": "error", "error": str(exc)})
            finally:
                progress.put(None)

        threading.Thread(target=worker, daemon=True).start()
        if not stream:
            item = progress.get()
            while item is not None:
                if item.get("status") == "success":
                    self._send_json(200, {"status": "success"})
                    return
                last = item
                item = progress.get()
            self._send_json(500, last or {"status": "error", "error": "pull failed"})
            return
        self._start_ndjson()
        heartbeat = time.monotonic()

        def beat() -> bool:
            nonlocal heartbeat
            now = time.monotonic()
            if now - heartbeat < 3.0:
                return False
            heartbeat = now
            return True

        try:
            while True:
                try:
                    item = progress.get(timeout=1.0)
                except queue.Empty:
                    continue
                if item is None:
                    break
                self._ndjson(item)
                if item.get("status") == "success":
                    break
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def api_delete(self) -> None:
        """DELETE /api/delete - remove an installed model from disk."""
        body_raw = self.read_body()
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        wanted = ""
        try:
            data = json.loads(body_raw.decode("utf-8") or "{}")
            if isinstance(data, dict):
                wanted = str(data.get("name") or data.get("model") or "")
        except ValueError:
            pass
        wanted = wanted or (query.get("name") or [""])[0]
        wanted = wanted.split(":")[0].strip()
        settings = self.manager.settings
        paths = [p for p in cb.list_models(settings)
                 if (cb.SHARD_RE.search(p.name).group(1) if cb.SHARD_RE.search(p.name)
                     else p.stem) == wanted]
        if not paths:
            self.send_error_reply(404, f"model '{wanted}' not found")
            return
        loaded_key = None
        if self.manager.model is not None:
            match = cb.SHARD_RE.search(self.manager.model.name)
            loaded_key = match.group(1) if match else self.manager.model.stem
        if loaded_key == wanted:
            self.manager.stop()  # unload before removing files
        removed = 0
        for path in paths:
            for victim in (path, path.with_suffix(".capybara.json")):
                try:
                    victim.unlink()
                    removed += 1
                except OSError:
                    pass
        self._send_json(200, {"status": "deleted" if removed else "not found",
                              "name": wanted})

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


class ApiError(Exception):
    """HTTP error destined for an API client."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def engine_stream(mgr: EngineManager, endpoint: str,
                  payload: Dict[str, Any]) -> Any:
    """POST to the engine's OpenAI endpoint and yield parsed SSE objects.

    Raises ApiError(503) when the engine is unreachable and ApiError with
    the engine's status/message on HTTP errors.
    """
    req = urllib.request.Request(
        f"{mgr.internal_url}{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=None)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace").strip()[:300]
        raise ApiError(exc.code or 502, detail or f"engine HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ApiError(503, f"engine unavailable: {exc}") from exc
    return resp


def iter_sse(resp: Any) -> Any:
    """Yield parsed JSON objects from a text/event-stream response."""
    with resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                yield json.loads(data)
            except ValueError:
                continue


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
        installed = sorted(cb.list_models(settings),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        if not installed:
            print(f"capybara: no models installed - try: capybara pull smollm",
                  file=sys.stderr)
            return 1
        model = installed[0]

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

    def install_signals() -> None:
        for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, shutdown)
            except (OSError, ValueError):
                pass  # not supported on this platform

    install_signals()

    keep_desc = "never unload" if mgr.keep_alive == math.inf else \
        f"unload after {mgr.keep_alive:.0f}s idle"
    print(f"Capybara {cb.VERSION} serving {mgr.model.name} "
          f"on http://{settings.host}:{settings.port} (UI: /, API: /v1, {keep_desc})")
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

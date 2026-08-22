#!/usr/bin/env python3
"""Capybara - a local model runner with an OpenAI-compatible API.

The CLI manages GGUF models (pull/list/rm/inspect), runs interactive chat
sessions and drives a llama.cpp `llama-server` process that exposes both a
native API and an OpenAI-compatible `/v1` endpoint.

Only the Python standard library is used.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VERSION = "0.3.0"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434
DEFAULT_CONTEXT = 10240
DEFAULT_BATCH = 2048
DEFAULT_UBATCH = 512
HF_API = "https://huggingface.co/api/models"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{file}"

SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
QUANT_SUFFIX_RE = re.compile(r"[-._](?:iq|q)\d[\w.\-]*$", re.IGNORECASE)

# Short names that expand to well-known Hugging Face GGUF repositories.
# Format: alias -> "owner/repo[:quant]".
ALIASES: Dict[str, str] = {
    "smollm": "tensorblock/SmolLM2-135M-Instruct-GGUF:Q2_K",
    "llama3": "bartowski/Meta-Llama-3-8B-Instruct-GGUF:Q4_K_M",
    "llama3.1": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M",
    "qwen2.5": "bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",
    "mistral": "bartowski/Mistral-7B-Instruct-v0.3-GGUF:Q4_K_M",
    "gemma2": "bartowski/gemma-2-9b-it-GGUF:Q4_K_M",
    "phi3": "microsoft/Phi-3-mini-4k-instruct-gguf:q4",
    "deepseek-r1": "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M",
}

DEFAULT_NUM_PREDICT = 2048

PARAM_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "repeat_penalty": "repeat_penalty",
    "seed": "seed",
    "num_predict": "max_tokens",
}


def die(msg: str) -> None:
    """Print an error to stderr and exit with status 1."""
    print(f"capybara: {msg}", file=sys.stderr)
    raise SystemExit(1)


def human_size(num: float) -> str:
    """Format a byte count as a human readable string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


class Settings:
    """Resolved runtime settings (defaults <- config file <- environment)."""

    def __init__(self, home: Path, cfg: Dict[str, Any]) -> None:
        env = os.environ.get

        def pick(env_key: str, *cfg_keys: str, default: Any) -> Any:
            if env_key in os.environ:
                return env(env_key)
            node: Any = cfg
            for key in cfg_keys:
                if not isinstance(node, dict) or key not in node:
                    return default
                node = node[key]
            return node

        self.home = home
        self.host = str(pick("CAPYBARA_HOST", "server", "host", default=DEFAULT_HOST))
        self.port = int(pick("CAPYBARA_PORT", "server", "port", default=DEFAULT_PORT))
        models_dir = pick("CAPYBARA_MODELS", "models", "directory", default=str(home / "models"))
        self.models = Path(models_dir).expanduser()
        threads = pick("CAPYBARA_THREADS", "runtime", "threads", default=os.cpu_count() or 4)
        self.threads = max(1, int(threads))
        self.context = int(pick("CAPYBARA_CONTEXT", "runtime", "context", default=DEFAULT_CONTEXT))
        self.batch = int(pick("CAPYBARA_BATCH", "runtime", "batch", default=DEFAULT_BATCH))
        self.ubatch = int(pick("CAPYBARA_UBATCH", "runtime", "ubatch", default=DEFAULT_UBATCH))
        gpu_layers = pick("CAPYBARA_GPU_LAYERS", "runtime", "gpu_layers", default=999)
        self.gpu_layers = int(gpu_layers)
        self.bin_dir = home / "bin"
        self.run_dir = home / "run"
        self.server_bin = self.bin_dir / "llama-server"
        self.log_file = self.run_dir / "server.log"
        self.state_file = self.run_dir / "server.json"

    @property
    def base_url(self) -> str:
        """Base URL of the managed llama-server."""
        return f"http://{self.host}:{self.port}"

    @property
    def openai_url(self) -> str:
        """OpenAI-compatible endpoint exposed by llama-server."""
        return f"{self.base_url}/v1"


def parse_scalar(raw: str) -> Any:
    """Convert a YAML-ish scalar string into bool/int/float/str."""
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    low = text.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def parse_config_text(text: str) -> Dict[str, Any]:
    """Parse a small YAML subset (nested maps of scalars) into a dict."""
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, sep, value = line.strip().partition(":")
        if not sep:
            continue
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        key = key.strip()
        value = value.strip()
        if value == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)
    return root


def load_settings(home: Optional[Path] = None) -> Settings:
    """Build Settings from defaults, config.yaml and environment variables."""
    home = Path(home or os.environ.get("CAPYBARA_HOME", str(Path.home() / ".capybara")))
    cfg: Dict[str, Any] = {}
    config_path = home / "config.yaml"
    if not config_path.exists():
        config_path = home / "config.yml"
    if config_path.exists():
        cfg = parse_config_text(config_path.read_text(encoding="utf-8"))
    return Settings(home, cfg)


def resolve_alias(spec: str) -> str:
    """Expand short model names like 'llama3' to HF repo specs."""
    return ALIASES.get(spec.lower(), spec)


def split_hf_spec(spec: str) -> Optional[Tuple[str, Optional[str]]]:
    """Split 'owner/repo[:quant]' into its parts; None when not an HF spec."""
    if spec.startswith(("http://", "https://")) or "/" not in spec:
        return None
    repo, sep, quant = spec.partition(":")
    if not sep:
        quant = None
    if "/" not in repo:
        return None
    return repo, quant or None


def select_gguf_files(names: List[str], quant: Optional[str] = None) -> List[str]:
    """Pick the best GGUF file(s) from a repository listing.

    Prefers files matching the requested quantization (default Q4_K_M),
    prefers single-file releases over sharded ones and returns every part
    of a shard set together so multi-part models download completely.
    """
    candidates = list(names)
    if quant:
        token = quant.lower()
        candidates = [n for n in candidates if token in n.lower()]
        if not candidates:
            return []
    preferred = [n for n in candidates if "q4_k_m" in n.lower()] or candidates
    singles = [n for n in preferred if not SHARD_RE.search(n)]
    if singles:
        return [sorted(singles, key=str.lower)[0]]
    groups: Dict[str, List[str]] = {}
    for name in preferred:
        match = SHARD_RE.search(name)
        if match:
            groups.setdefault(match.group(1), []).append(name)
    if not groups:
        return []
    base = sorted(groups)[0]
    return sorted(groups[base])


def http_json(url: str, timeout: float = 20.0) -> Any:
    """GET a URL and decode its JSON body; raises on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": "capybara"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def hf_repo_ggufs(repo: str) -> List[str]:
    """List GGUF filenames published in a Hugging Face repository."""
    info = http_json(f"{HF_API}/{repo}")
    siblings = [s.get("rfilename", "") for s in info.get("siblings", [])]
    return [s for s in siblings if s.lower().endswith(".gguf")]


def download_to(dest: Path, url: str) -> None:
    """Download url to dest with curl when available, urllib otherwise."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl")
    if curl:
        cmd = [curl, "-L", "--fail", "--retry", "4", "--retry-delay", "2",
               "--progress-bar", "-o", str(dest), url]
        if subprocess.run(cmd).returncode != 0:
            dest.unlink(missing_ok=True)
            die(f"download failed: {url}")
        return
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "capybara"})
        with urllib.request.urlopen(req) as resp, open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        die(f"download failed: {exc}")


def clone_file(src: Path, dest: Path) -> None:
    """Clone src to dest via hardlink when possible, falling back to a copy."""
    dest.unlink(missing_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def list_models(settings: Settings) -> List[Path]:
    """All installed GGUF files, newest first."""
    settings.models.mkdir(parents=True, exist_ok=True)
    return sorted(settings.models.glob("*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True)


def resolve_model(settings: Settings, name: str) -> Optional[Path]:
    """Resolve a model reference to a local file, if one exists."""
    direct = Path(name).expanduser()
    if direct.exists():
        return direct
    exact = settings.models / name
    if exact.exists():
        return exact
    stem = name[:-5] if name.lower().endswith(".gguf") else name
    wanted = stem.lower()
    matches: List[Path] = []
    for path in list_models(settings):
        file_stem = path.stem.lower()
        base = QUANT_SUFFIX_RE.sub("", file_stem)
        base = re.sub(r"[-._]\d{5}-of-\d{5}$", "", base)
        if file_stem == wanted or path.name == name or base == wanted:
            matches.append(path)
    if not matches:
        return None
    first_parts = [p for p in matches
                   if SHARD_RE.search(p.name) and SHARD_RE.search(p.name).group(2) == "00001"]
    return first_parts[0] if first_parts else matches[0]


def pull_model(settings: Settings, spec: str) -> Path:
    """Install a model from a local path, URL or Hugging Face repo[:quant]."""
    spec = resolve_alias(spec)
    existing = resolve_model(settings, spec)
    parsed = split_hf_spec(spec)

    if existing is not None and parsed is None:
        print(f"already installed: {existing}")
        return existing
    settings.models.mkdir(parents=True, exist_ok=True)

    if spec.startswith(("http://", "https://")):
        filename = Path(urllib.parse.urlparse(spec).path).name or "model.gguf"
        if not filename.lower().endswith(".gguf"):
            filename += ".gguf"
        dest = settings.models / filename
        download_to(dest, spec)
        print(f"installed {dest}")
        return dest

    local = Path(spec).expanduser()
    if local.exists():
        dest = settings.models / local.name
        clone_file(local, dest)
        print(f"installed {dest}")
        return dest

    if parsed:
        repo, quant = parsed
        try:
            ggufs = hf_repo_ggufs(repo)
        except Exception as exc:
            die(f"Hugging Face lookup failed for {repo}: {exc}")
        if not ggufs:
            die(f"no GGUF files found in {repo}")
        chosen = select_gguf_files(ggufs, quant)
        if not chosen:
            die(f"no GGUF matching quantization '{quant}' in {repo}")
        for filename in chosen:
            dest = settings.models / filename
            if dest.exists() and dest.stat().st_size > 0:
                print(f"already installed {dest}")
                continue
            url = HF_RESOLVE.format(repo=repo, file=urllib.parse.quote(filename))
            download_to(dest, url)
            print(f"installed {dest}")
        return settings.models / chosen[-1]

    known = ", ".join(sorted(ALIASES))
    die(f"unknown model '{spec}' - use a local GGUF path, a URL, owner/repo[:quant], "
        f"or an alias ({known})")


def hf_spec_local_files(settings: Settings, spec: str) -> List[Path]:
    """Local files that correspond to an HF 'owner/repo[:quant]' spec."""
    parsed = split_hf_spec(spec)
    if not parsed:
        return []
    repo, quant = parsed
    try:
        chosen = select_gguf_files(hf_repo_ggufs(repo), quant)
    except Exception:
        return []
    return [f for f in (settings.models / name for name in chosen) if f.exists()]


def resolve_model_or_alias(settings: Settings, name: str) -> Optional[Path]:
    """Resolve a model reference, expanding known aliases first."""
    found = resolve_model(settings, name)
    if found:
        return found
    for spec in (name, resolve_alias(name)):
        files = hf_spec_local_files(settings, spec)
        if files:
            return files[-1]
    return None


def sidecar_for(model: Path) -> Optional[Dict[str, Any]]:
    """Load the metadata JSON written by 'capybara create', if present."""
    meta = model.with_suffix(".capybara.json")
    if not meta.exists():
        return None
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def parse_modelfile(text: str) -> Dict[str, Any]:
    """Parse an Ollama-style Modelfile into base/system/template/params."""
    result: Dict[str, Any] = {"base": None, "system": None, "template": None, "params": {}}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z]+)\s*(.*)$", line)
        if not match:
            continue
        directive = match.group(1).upper()
        rest = match.group(2).strip()
        if directive == "FROM":
            result["base"] = rest
        elif directive == "PARAMETER":
            name, _, value = rest.partition(" ")
            result["params"][name.lower()] = parse_scalar(value.strip())
        elif directive in ("SYSTEM", "TEMPLATE"):
            if rest.startswith('"""'):
                block = rest[3:]
                if block.endswith('"""') and len(block) >= 3:
                    block = block[:-3]
                elif not rest.endswith('"""'):
                    closing = rest[3:] + "\n"
                    while index < len(lines):
                        row = lines[index]
                        index += 1
                        end = row.find('"""')
                        if end != -1:
                            closing += row[:end]
                            break
                        closing += row + "\n"
                    block = closing
                result[directive.lower()] = block.strip("\n")
            elif rest:
                result[directive.lower()] = rest
    return result


def request_context(model: Path) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """System message + sampling overrides contributed by a model's sidecar."""
    meta = sidecar_for(model) or {}
    messages: List[Dict[str, str]] = []
    if meta.get("system"):
        messages.append({"role": "system", "content": str(meta["system"])})
    params_in = meta.get("params") or {}
    payload_params = {PARAM_MAP[k]: v for k, v in params_in.items() if k in PARAM_MAP}
    # Guard against runaway generation unless the model sets num_predict.
    payload_params.setdefault("max_tokens", DEFAULT_NUM_PREDICT)
    return messages, payload_params


def server_state(settings: Settings) -> Optional[Dict[str, Any]]:
    """Read the state file describing the managed server process."""
    if not settings.state_file.exists():
        return None
    try:
        data = json.loads(settings.state_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def server_ready(settings: Settings, timeout: float = 2.0) -> bool:
    """True when the managed server answers /health with HTTP 200."""
    try:
        req = urllib.request.Request(f"{settings.base_url}/health")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


def stop_server(settings: Settings) -> None:
    """Terminate the managed server process and clear its state."""
    state = server_state(settings)
    pid_file = settings.run_dir / "server.pid"
    pid = None
    if state and state.get("pid"):
        pid = int(state["pid"])
    elif pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
        except ValueError:
            pid = None
    if pid is None:
        print("Capybara is not running")
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(40):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    pid_file.unlink(missing_ok=True)
    settings.state_file.unlink(missing_ok=True)
    print("Capybara stopped")


def start_server(settings: Settings, model: Path, foreground: bool = False,
                 extra_args: Optional[List[str]] = None) -> None:
    """Start llama-server for model, restarting it if another model is loaded."""
    if not settings.server_bin.exists():
        die(f"engine not found at {settings.server_bin} - run ./install.sh first")
    state = server_state(settings)
    loaded = state.get("model") if state else None
    if server_ready(settings) and loaded == model.name:
        return
    if server_ready(settings):
        stop_server(settings)
    settings.run_dir.mkdir(parents=True, exist_ok=True)

    meta = sidecar_for(model) or {}
    ctx = int((meta.get("params") or {}).get("num_ctx", settings.context))
    cmd = [
        str(settings.server_bin),
        "--model", str(model),
        "--host", settings.host,
        "--port", str(settings.port),
        "--threads", str(settings.threads),
        "--ctx-size", str(ctx),
        "--batch-size", str(settings.batch),
        "--ubatch-size", str(settings.ubatch),
        "--n-gpu-layers", str(settings.gpu_layers),
        "--parallel", "1",
        "--cont-batching",
        "--flash-attn", "on",
    ]
    if extra_args:
        cmd.extend(extra_args)

    if foreground:
        print(f"Capybara serving {model.name} on {settings.base_url} (Ctrl-C to stop)")
        try:
            subprocess.run(cmd, check=False)
        finally:
            settings.state_file.unlink(missing_ok=True)
        return

    with open(settings.log_file, "ab") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=log)
    (settings.run_dir / "server.pid").write_text(str(proc.pid))
    for _ in range(120):
        if server_ready(settings, timeout=1.0):
            settings.state_file.write_text(json.dumps({
                "pid": proc.pid,
                "model": model.name,
                "host": settings.host,
                "port": settings.port,
                "started_at": time.time(),
            }))
            return
        if proc.poll() is not None:
            break
        time.sleep(0.25)
    tail = ""
    if settings.log_file.exists():
        lines = settings.log_file.read_text(errors="replace").splitlines()
        tail = "\n".join(lines[-10:])
    stop_server(settings)
    die(f"server failed to start; see {settings.log_file}\n{tail}")


def stream_chat_completions(settings: Settings, payload: Dict[str, Any]) -> str:
    """POST to /v1/chat/completions streaming deltas to stdout; returns text."""
    req = urllib.request.Request(
        settings.openai_url + "/chat/completions",
        data=json.dumps({**payload, "stream": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks: List[str] = []
    with urllib.request.urlopen(req, timeout=None) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                delta = json.loads(body)["choices"][0]["delta"].get("content") or ""
            except (KeyError, IndexError, ValueError):
                continue
            if delta:
                chunks.append(delta)
                print(delta, end="", flush=True)
    print(flush=True)
    return "".join(chunks)


def chat_payload(model_name: str, history: List[Dict[str, str]],
                 params: Dict[str, Any]) -> Dict[str, Any]:
    """Build the request body for the OpenAI-compatible endpoint."""
    return {"model": model_name, "messages": history, **params}


def ensure_server(settings: Settings, model: Path) -> None:
    """Make sure the server is running with exactly this model loaded."""
    start_server(settings, model)


def interactive_chat(settings: Settings, model: Path) -> None:
    """Run a REPL chat session against the given model."""
    system_msgs, params = request_context(model)
    history: List[Dict[str, str]] = list(system_msgs)
    print(f"Capybara running {model.name}. Type /bye to exit, /clear to reset.")
    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        prompt = prompt.strip()
        if prompt in ("/bye", "/exit", "/quit"):
            return
        if prompt == "/clear":
            history = list(system_msgs)
            print("context cleared")
            continue
        if prompt == "/help":
            print("/bye exit | /clear reset conversation")
            continue
        if not prompt:
            continue
        history.append({"role": "user", "content": prompt})
        try:
            reply = stream_chat_completions(
                settings, chat_payload(model.name, history, params))
            history.append({"role": "assistant", "content": reply})
        except SystemExit:
            raise
        except Exception as exc:
            print(f"capybara: {exc}")


def generate_once(settings: Settings, model: Path, prompt: str) -> None:
    """Send a single prompt and stream the completion to stdout."""
    system_msgs, params = request_context(model)
    history = system_msgs + [{"role": "user", "content": prompt}]
    stream_chat_completions(settings, chat_payload(model.name, history, params))


def group_models(paths: List[Path]) -> List[Tuple[str, int]]:
    """Collapse shard sets into single logical entries (base, total bytes)."""
    entries: Dict[str, int] = {}
    for path in paths:
        match = SHARD_RE.search(path.name)
        key = match.group(1) if match else path.stem
        entries[key] = entries.get(key, 0) + path.stat().st_size
    return sorted(entries.items())


def do_list(settings: Settings) -> None:
    """Handle the 'list' command."""
    rows = group_models(list_models(settings))
    if not rows:
        print("no models installed - try: capybara pull smollm")
        return
    width = max(len(name) for name, _ in rows)
    print(f"{'NAME':<{width}}  {'SIZE':>9}  MODIFIED")
    seen: set[str] = set()
    for path in list_models(settings):
        match = SHARD_RE.search(path.name)
        key = match.group(1) if match else path.stem
        if key in seen:
            continue
        seen.add(key)
        size = next(s for n, s in rows if n == key)
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))
        print(f"{key:<{width}}  {human_size(size):>9}  {stamp}")


def do_show(settings: Settings, name: str) -> None:
    """Handle the 'show'/'inspect' commands."""
    model = resolve_model_or_alias(settings, name)
    if model is None:
        die(f"model not found: {name}")
    meta = sidecar_for(model) or {}
    print(f"Name:     {model.stem}")
    print(f"Path:     {model}")
    print(f"Size:     {human_size(model.stat().st_size)} ({model.stat().st_size:,} bytes)")
    shard = SHARD_RE.search(model.name)
    if shard:
        base = shard.group(1)
        parts = [p.name for p in list_models(settings)
                 if SHARD_RE.search(p.name) and SHARD_RE.search(p.name).group(1) == base]
        print(f"Shards:   {len(parts)}")
    if meta.get("base"):
        print(f"Base:     {meta['base']}")
    params = meta.get("params") or {}
    if params:
        pretty = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
        print(f"Params:   {pretty}")
    if meta.get("system"):
        preview = str(meta["system"]).replace("\n", " ")[:80]
        print(f"System:   {preview}...")
    engine = settings.server_bin
    if engine.exists():
        out = subprocess.run([str(engine), "--version"], capture_output=True, text=True)
        version_line = (out.stdout or out.stderr).strip().splitlines()
        if version_line:
            print(f"Engine:   llama.cpp ({version_line[0]})")


def do_create(settings: Settings, name: str, modelfile: str) -> None:
    """Handle the 'create' command: clone a base model plus Modelfile config."""
    parsed = parse_modelfile(Path(modelfile).read_text(encoding="utf-8"))
    if not parsed["base"]:
        die("Modelfile needs a FROM directive")
    base = resolve_model(settings, parsed["base"])
    if base is None:
        die(f"base model not installed: {parsed['base']}")
    out_name = name if name.lower().endswith(".gguf") else name + ".gguf"
    dest = settings.models / out_name
    clone_file(base, dest)
    meta = {
        "name": name,
        "base": parsed["base"],
        "system": parsed["system"],
        "template": parsed["template"],
        "params": parsed["params"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    dest.with_suffix(".capybara.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"created {dest}")


def do_ps(settings: Settings) -> None:
    """Handle the 'ps' command."""
    state = server_state(settings)
    if not server_ready(settings) or not state:
        print("STOPPED")
        return
    uptime = time.time() - float(state.get("started_at", time.time()))
    print(f"RUNNING  model={state.get('model')}  pid={state.get('pid')}  "
          f"url={settings.base_url}  uptime={int(uptime)}s")


def do_logs(settings: Settings, lines: int) -> None:
    """Handle the 'logs' command."""
    if not settings.log_file.exists():
        print("No logs yet.")
        return
    content = settings.log_file.read_text(errors="replace").splitlines()
    print("\n".join(content[-lines:]))


def do_serve(settings: Settings, args: argparse.Namespace) -> None:
    """Handle the 'serve' command."""
    spec = args.model or os.environ.get("CAPYBARA_MODEL")
    model = resolve_model(settings, spec) if spec else None
    if model is None:
        installed = list_models(settings)
        if not installed:
            die("no models installed - try: capybara pull smollm")
        model = installed[0]
    start_server(settings, model, foreground=args.foreground)
    if not args.foreground:
        print(f"Capybara serving {model.name}")
        print(f"OpenAI API: {settings.openai_url}/chat/completions")
        print(f"Health:     {settings.base_url}/health")


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="capybara", description=__doc__)
    parser.add_argument("--version", action="version", version=f"Capybara {VERSION}")
    sub = parser.add_subparsers(dest="cmd")

    serve = sub.add_parser("serve", help="start the API server")
    serve.add_argument("--model", help="model to load")
    serve.add_argument("-F", "--foreground", action="store_true",
                       help="run the engine in the foreground")

    run = sub.add_parser("run", help="run a model (interactive chat or one-shot)")
    run.add_argument("model")
    run.add_argument("prompt", nargs="?")

    pull = sub.add_parser("pull", help="download a model")
    pull.add_argument("model")

    sub.add_parser("list", aliases=["ls"], help="list installed models")

    show = sub.add_parser("show", help="show details about a model")
    show.add_argument("model")
    inspect = sub.add_parser("inspect", help="alias for show")
    inspect.add_argument("model")

    rm = sub.add_parser("rm", help="remove a model")
    rm.add_argument("model")

    cp = sub.add_parser("cp", help="copy a model under a new name")
    cp.add_argument("source")
    cp.add_argument("destination")

    create = sub.add_parser("create", help="create a model from a Modelfile")
    create.add_argument("-f", "--file", default="Modelfile")
    create.add_argument("name")

    sub.add_parser("ps", help="show server status")
    sub.add_parser("stop", help="stop the API server")
    sub.add_parser("serve-status", help=argparse.SUPPRESS)

    logs = sub.add_parser("logs", help="show engine logs")
    logs.add_argument("-n", type=int, default=50, help="number of lines (default 50)")

    generate = sub.add_parser("generate", help="one-shot generation from a prompt")
    generate.add_argument("model")
    generate.add_argument("prompt")

    launch = sub.add_parser("launch", help="launch a program wired to the API")
    launch.add_argument("integration")
    launch.add_argument("--model")

    sub.add_parser("version", help="print version")
    sub.add_parser("help", help="show help")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd in (None, "help"):
        parser.print_help()
        return
    settings = load_settings()
    if args.cmd == "version":
        print(f"Capybara {VERSION}")
    elif args.cmd == "serve":
        do_serve(settings, args)
    elif args.cmd == "run":
        model = resolve_model_or_alias(settings, args.model)
        if model is None:
            model = pull_model(settings, args.model)
        ensure_server(settings, model)
        if args.prompt:
            generate_once(settings, model, args.prompt)
        else:
            interactive_chat(settings, model)
    elif args.cmd == "pull":
        pull_model(settings, args.model)
    elif args.cmd in ("list", "ls"):
        do_list(settings)
    elif args.cmd in ("show", "inspect"):
        do_show(settings, args.model)
    elif args.cmd == "rm":
        model = resolve_model_or_alias(settings, args.model)
        if model is None:
            die(f"model not found: {args.model}")
        model.unlink(missing_ok=True)
        model.with_suffix(".capybara.json").unlink(missing_ok=True)
        print(f"deleted {args.model}")
    elif args.cmd == "cp":
        src = resolve_model_or_alias(settings, args.source)
        if src is None:
            die(f"source model not found: {args.source}")
        dst_name = args.destination
        if not dst_name.lower().endswith(".gguf"):
            dst_name += ".gguf"
        dest = settings.models / dst_name
        clone_file(src, dest)
        print(f"copied {args.source} -> {dest.name}")
    elif args.cmd == "create":
        do_create(settings, args.name, args.file)
    elif args.cmd == "ps":
        do_ps(settings)
    elif args.cmd == "stop":
        stop_server(settings)
    elif args.cmd == "logs":
        do_logs(settings, args.n)
    elif args.cmd == "generate":
        model = resolve_model_or_alias(settings, args.model)
        if model is None:
            model = pull_model(settings, args.model)
        ensure_server(settings, model)
        generate_once(settings, model, args.prompt)
    elif args.cmd == "launch":
        exe = shutil.which(args.integration)
        if not exe:
            die(f"{args.integration} not installed")
        env = os.environ.copy()
        env["OPENAI_BASE_URL"] = settings.openai_url
        env["OPENAI_API_BASE"] = settings.openai_url
        env["OPENAI_API_KEY"] = "capybara"
        if args.model:
            env["CAPYBARA_MODEL"] = args.model
        subprocess.run([exe], env=env, check=False)
    else:
        die(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

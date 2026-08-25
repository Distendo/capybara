#!/usr/bin/env python3
"""Run GGUF models locally via llama.cpp.

Single-file CLI + gateway. No external backend.
"""

import argparse
import json
import math
import os
import queue
import re
import shutil
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
from typing import Any, Dict, List, Optional, Tuple

IS_WINDOWS = os.name == "nt"
EXE = ".exe" if IS_WINDOWS else ""

VERSION = "1.3.1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434
DEFAULT_WEBUI_PORT = 8080
DEFAULT_CONTEXT = 10240
DEFAULT_BATCH = 2048
DEFAULT_UBATCH = 512
DEFAULT_KEEP_ALIVE = "5m"
HF_API = "https://huggingface.co/api/models"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{file}"
OLLAMA_REGISTRY = "https://registry.ollama.ai"

# Ollama registry layers (OCI media types).
OLLAMA_MODEL_MT = "application/vnd.ollama.image.model"
OLLAMA_SYSTEM_MT = "application/vnd.ollama.image.system"
OLLAMA_PARAMS_MT = "application/vnd.ollama.image.params"
OLLAMA_MANIFEST_ACCEPT = ", ".join([
    "application/vnd.ollama.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
])

SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
QUANT_SUFFIX_RE = re.compile(r"[-._](?:iq|q)\d[\w.\-]*$", re.IGNORECASE)

DEFAULT_NUM_PREDICT = 2048

PARAM_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "repeat_penalty": "repeat_penalty",
    "seed": "seed",
    "num_predict": "max_tokens",
    "stop": "stop",
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
        self.webui_port = int(pick("CAPYBARA_WEBUI_PORT", "server", "webui_port",
                                   default=DEFAULT_WEBUI_PORT))
        models_dir = pick("CAPYBARA_MODELS", "models", "directory", default=str(home / "models"))
        self.models = Path(models_dir).expanduser()
        threads = pick("CAPYBARA_THREADS", "runtime", "threads", default=os.cpu_count() or 4)
        self.threads = max(1, int(threads))
        self.context = int(pick("CAPYBARA_CONTEXT", "runtime", "context", default=DEFAULT_CONTEXT))
        self.batch = int(pick("CAPYBARA_BATCH", "runtime", "batch", default=DEFAULT_BATCH))
        self.ubatch = int(pick("CAPYBARA_UBATCH", "runtime", "ubatch", default=DEFAULT_UBATCH))
        gpu_layers = pick("CAPYBARA_GPU_LAYERS", "runtime", "gpu_layers", default=999)
        self.gpu_layers = int(gpu_layers)
        self.keep_alive_raw = str(pick("CAPYBARA_KEEP_ALIVE", "runtime", "keep_alive",
                                       default=DEFAULT_KEEP_ALIVE))
        self.bin_dir = home / "bin"
        self.run_dir = home / "run"
        self.server_bin = self._resolve_engine(pick(
            "CAPYBARA_ENGINE", "runtime", "engine", default=None))
        self.log_file = self.run_dir / "server.log"
        self.state_file = self.run_dir / "server.json"

    def _resolve_engine(self, configured: Any) -> Path:
        """Locate llama-server: config path > app dir > install dir > PATH."""
        candidates: List[Path] = []
        if configured:
            candidates.append(Path(str(configured)).expanduser())
        # Next to this script - covers portable installs where the engine
        # ships inside the same folder tree as capybara.py.
        candidates.append(Path(__file__).resolve().parent / "bin"
                          / f"llama-server{EXE}")
        candidates.append(self.home / "bin" / f"llama-server{EXE}")
        found = shutil.which("llama-server")
        if found:
            candidates.append(Path(found))
        for cand in candidates:
            if cand.exists():
                return cand
        return candidates[0]

    @property
    def engine_port(self) -> int:
        """Internal port the engine binds to (gateway owns the public one)."""
        return self.port + 1

    @property
    def base_url(self) -> str:
        """Base URL of the managed llama-server."""
        return f"http://{self.host}:{self.port}"

    @property
    def openai_url(self) -> str:
        """OpenAI-compatible endpoint exposed by llama-server."""
        return f"{self.base_url}/v1"

    @property
    def server_py(self) -> Path:
        """This file itself acts as the gateway when called with the `gateway` subcommand."""
        return Path(__file__).resolve()


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


def parse_keep_alive(value: Any, default: float = 300.0) -> float:
    """Parse an Ollama-style duration into seconds.

    Accepts numbers (seconds) or strings like "90", "90s", "5m", "2h",
    compound forms such as "1h30m", and the specials "-1"/"0" (never
    unload / unload right after each response). Invalid input falls back
    to ``default``.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float("inf") if value < 0 else float(value)
    text = str(value).strip().lower()
    if not text:
        return default
    try:
        num = float(text)
        return float("inf") if num < 0 else num
    except ValueError:
        pass
    units = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    total = 0.0
    for number, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(ms|[smh])?", text):
        total += float(number) * units.get(unit or "s", 1.0)
    return total if total > 0 else default


# Ollama request 'options' -> llama.cpp OpenAI fields. Keys mapped to None
# are accepted but cannot be applied to a running engine (ctx is fixed at
# spawn time); anything outside this table is dropped.
OLLAMA_OPTIONS_MAP: Dict[str, Optional[str]] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "repeat_penalty": "repeat_penalty",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "seed": "seed",
    "num_predict": "max_tokens",
    "max_tokens": "max_tokens",
    "stop": "stop",
    "num_ctx": None,
    "num_batch": None,
    "num_gpu": None,
    "mirostat": None,
    "mirostat_eta": None,
    "mirostat_tau": None,
    "tfs_z": None,
    "typical_p": None,
    "num_keep": None,
    "penalize_newline": None,
}


def ollama_options_to_openai(options: Any) -> Dict[str, Any]:
    """Translate an Ollama request's options dict into engine parameters."""
    if not isinstance(options, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, value in options.items():
        target = OLLAMA_OPTIONS_MAP.get(str(key), False)
        if target and value is not None:
            out[target] = value
    # Ollama sends stop as a string; engines expect a list.
    if isinstance(out.get("stop"), str):
        out["stop"] = [out["stop"]]
    return out


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
    """Return the spec unchanged (aliases removed)."""
    return spec


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


def require_free_space(dest: Path, needed: int) -> None:
    """Refuse to fill the disk: die when free space < needed * 1.05."""
    try:
        free = shutil.disk_usage(dest).free
    except OSError:
        return
    required = int(needed * 1.05)
    if free < required:
        die(f"not enough disk space for {dest.name}: need ~{human_size(required)}, "
            f"only {human_size(free)} free - free up some space and retry")


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

    ollama_ref = parse_ollama_ref(spec)
    if ollama_ref is not None:
        name, tag = ollama_ref
        explicit = bool(re.match(r"^(?:ollama|ol)/", spec))
        imported = import_ollama_local(settings, name, tag)
        if imported:
            return imported
        try:
            return pull_ollama_registry(settings, name, tag)
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and explicit:
                die(f"model '{name}:{tag}' not found in the Ollama registry")
            if exc.code != 404:
                die(f"Ollama registry lookup failed for {name}:{tag}: {exc}")
        except urllib.error.URLError as exc:
            die(f"Ollama registry unreachable ({exc.reason}) - check your connection")

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

    die(f"unknown model '{spec}' - use a local GGUF path, a URL, owner/repo[:quant],\n"
        f"or an Ollama library model (e.g. ollama/gemma3, ollama/phi4)")


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


def parse_ollama_ref(spec: str) -> Optional[Tuple[str, str]]:
    """Split an Ollama-style ref into (name, tag).

    Accepts explicit 'ollama/name[:tag]' / 'ol/name[:tag]' prefixes and any
    bare 'name[:tag]' without a slash (implicit registry candidates).
    Returns None for URLs and owner/repo specs.
    """
    if spec.startswith(("http://", "https://")):
        return None
    explicit = re.match(r"^(?:ollama|ol)/([^/:]+)(?::(.+))?$", spec)
    if explicit:
        return explicit.group(1).lower(), explicit.group(2) or "latest"
    if "/" in spec:
        return None
    name, _, tag = spec.partition(":")
    if not name:
        return None
    return name.lower(), tag or "latest"


def ollama_manifest(name: str, tag: str) -> Dict[str, Any]:
    """Fetch an OCI manifest from the Ollama registry."""
    url = f"{OLLAMA_REGISTRY}/v2/library/{name}/manifests/{tag}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "capybara",
        "Accept": OLLAMA_MANIFEST_ACCEPT,
    })
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def local_ollama_root(root: Optional[Path] = None) -> Optional[Path]:
    """The models directory of a local Ollama installation, if present."""
    if root is None:
        env = os.environ.get("OLLAMA_MODELS")
        root = Path(env).expanduser() if env else Path.home() / ".ollama" / "models"
    return root if root.is_dir() else None


def ollama_gguf_name(name: str, tag: str, parts: int = 1) -> List[str]:
    """Local filename(s) for a pulled Ollama model; tags become suffixes,
    multi-blob models get llama.cpp shard names so they load together."""
    base = f"{name}-{tag}" if tag != "latest" else name
    if parts == 1:
        return [f"{base}.gguf"]
    return [f"{base}-{i:05d}-of-{parts:05d}.gguf" for i in range(1, parts + 1)]


def install_ollama_manifest(settings: Settings, name: str, tag: str,
                            manifest: Dict[str, Any],
                            fetch_blob, blob_text) -> Path:
    """Install the GGUF blob(s) of an Ollama manifest plus its sidecar.

    fetch_blob(digest, dest) materializes one layer;
    blob_text(digest) returns small text layers (system/params).
    """
    settings.models.mkdir(parents=True, exist_ok=True)
    layers = manifest.get("layers", [])
    media_types = {l.get("mediaType") for l in layers}
    model_layers = [l for l in layers if l.get("mediaType") == OLLAMA_MODEL_MT]
    if not model_layers:
        if "application/vnd.ollama.image.tensor" in media_types:
            die(f"{name}:{tag} is not a GGUF model (MLX/safetensors build) - "
                f"capybara only runs GGUF models")
        die(f"{name}:{tag} contains no model data")
    filenames = ollama_gguf_name(name, tag, len(model_layers))
    dests: List[Path] = []
    for filename, layer in zip(filenames, model_layers):
        dest = settings.models / filename
        size = int(layer.get("size") or 0)
        if not (dest.exists() and dest.stat().st_size > 0):
            if size:
                require_free_space(dest, size)
            fetch_blob(layer["digest"], dest)
            with open(dest, "rb") as fh:
                magic = fh.read(4)
            if magic != b"GGUF":
                dest.unlink(missing_ok=True)
                die(f"{name}:{tag} is not a GGUF model "
                    f"(unsupported format) - pick a GGUF variant")
            print(f"installed {dest}")
        else:
            print(f"already installed {dest}")
        dests.append(dest)
    sidecar: Dict[str, Any] = {"template": None}
    for layer in layers:
        media = layer.get("mediaType")
        digest = layer.get("digest")
        if not digest:
            continue
        try:
            text = blob_text(digest)
        except OSError:
            continue
        if media == OLLAMA_SYSTEM_MT and text.strip():
            sidecar["system"] = text
        elif media == OLLAMA_PARAMS_MT:
            try:
                sidecar["params"] = {k: v for k, v in json.loads(text).items()
                                     if k in PARAM_MAP}
            except ValueError:
                pass
    if "system" in sidecar or sidecar.get("params"):
        sidecar.update({"name": Path(filenames[0]).stem,
                        "base": f"ollama:{name}:{tag}",
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        meta = settings.models / f"{Path(filenames[0]).stem}.capybara.json"
        meta.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return dests[-1]


def pull_ollama_registry(settings: Settings, name: str, tag: str) -> Path:
    """Download a model straight from the Ollama registry."""
    manifest = ollama_manifest(name, tag)

    def fetch_blob(digest: str, dest: Path) -> None:
        download_to(dest, f"{OLLAMA_REGISTRY}/v2/library/{name}/blobs/{digest}")

    def blob_text(digest: str) -> str:
        req = urllib.request.Request(
            f"{OLLAMA_REGISTRY}/v2/library/{name}/blobs/{digest}",
            headers={"User-Agent": "capybara"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="replace")

    result = install_ollama_manifest(settings, name, tag, manifest,
                                     fetch_blob, blob_text)
    print(f"(from Ollama registry: {name}:{tag})")
    return result


def import_ollama_local(settings: Settings, name: str, tag: str,
                        root: Optional[Path] = None) -> Optional[Path]:
    """Import a model already pulled by a local Ollama install.

    Blobs are hardlinked when possible, so this is instant and free.
    Returns None when Ollama is absent or does not know the model.
    """
    source = local_ollama_root(root)
    if source is None:
        return None
    mpath = (source / "manifests" / "registry.ollama.ai" /
             "library" / name / tag)
    if not mpath.is_file():
        return None
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except ValueError as exc:
        die(f"unreadable Ollama manifest {mpath}: {exc}")

    def blob_path(digest: str) -> Path:
        return source / "blobs" / digest.replace(":", "-")

    def fetch_blob(digest: str, dest: Path) -> None:
        src = blob_path(digest)
        if not src.is_file():
            die(f"Ollama blob missing: {src}")
        clone_file(src, dest)

    def blob_text(digest: str) -> str:
        path = blob_path(digest)
        return path.read_text(encoding="utf-8", errors="replace")

    result = install_ollama_manifest(settings, name, tag, manifest,
                                     fetch_blob, blob_text)
    print(f"(imported from local Ollama at {source})")
    return result


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
    """Read the state file describing the managed gateway/engine processes."""
    if not settings.state_file.exists():
        return None
    try:
        data = json.loads(settings.state_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def process_comm(pid: int) -> str:
    """Return the command name of a pid ('' when it does not exist)."""
    try:
        if IS_WINDOWS:
            # os.kill(pid, 0) terminates on Windows, so probe via tasklist.
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=10)
            line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
            return line.split('","')[0].strip('"') if line.startswith('"') else ""
        out = subprocess.run(["ps", "-p", str(pid), "-o", "comm="],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return ""


def process_matches(pid: int, *needles: str) -> bool:
    """True when a pid is alive and its command contains one of the needles.

    Protects against killing an innocent process whose numeric id was
    recycled after Capybara crashed without cleaning up.
    """
    comm = process_comm(pid)
    low = comm.lower()
    return bool(comm) and any(n in low for n in needles)


def state_is_ours(settings: Settings, state: Optional[Dict[str, Any]] = None) -> bool:
    """True when the recorded gateway process is still alive and is ours."""
    state = state or server_state(settings)
    if not state:
        return False
    pid = state.get("gateway_pid") or state.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    return process_matches(pid, "python", "capybara")


def server_ready(settings: Settings, timeout: float = 2.0) -> bool:
    """True when the managed gateway answers /health with HTTP 200."""
    try:
        req = urllib.request.Request(f"{settings.base_url}/health")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


def port_has_listener(settings: Settings, port: Optional[int] = None) -> bool:
    """True when anything accepts TCP connections on host:port."""
    import socket
    target = settings.port if port is None else port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((settings.host, target)) == 0


def foreign_port_guard(settings: Settings) -> None:
    """Die with a clear message when the port is held by another program."""
    if not port_has_listener(settings):
        return
    if server_ready(settings) and state_is_ours(settings):
        return
    die(f"port {settings.port} is already used by another program "
        f"(is Ollama or an old Capybara running?).\n"
        f"Stop it or choose another port, e.g.:\n"
        f"  export CAPYBARA_PORT=11440")


def stop_server(settings: Settings) -> None:
    """Terminate the gateway (and its engine child) and clear state."""
    state = server_state(settings)
    stopped_any = False
    sig_kill = signal.SIGTERM if IS_WINDOWS else signal.SIGKILL
    for key, needles in (("gateway_pid", ("python", "capybara")),
                         ("pid", ("llama-server",))):
        pid = state.get(key) if state else None
        if not isinstance(pid, int) or pid <= 0:
            continue
        if not process_matches(pid, *needles):
            continue  # stale entry pointing at an unrelated process
        try:
            os.kill(pid, signal.SIGTERM)
            stopped_any = True
        except OSError:
            pass
        for _ in range(50):
            # os.kill(pid, 0) kills on Windows - probe the name instead.
            if not process_comm(pid):
                break
            time.sleep(0.1)
        else:
            try:
                os.kill(pid, sig_kill)
            except OSError:
                pass
    (settings.run_dir / "server.pid").unlink(missing_ok=True)
    settings.state_file.unlink(missing_ok=True)
    if not stopped_any:
        print("Capybara is not running")
    else:
        print("Capybara stopped")


def wait_until_ready(settings: Settings, proc: subprocess.Popen,
                     seconds: float = 180.0) -> None:
    """Block until the freshly spawned gateway serves /health."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        if server_ready(settings, timeout=1.0):
            time.sleep(0.3)
            return
        time.sleep(0.25)
    tail = ""
    gw_log = settings.run_dir / "gateway.log"
    for log in (gw_log, settings.log_file):
        if log.exists():
            lines = log.read_text(errors="replace").splitlines()
            tail += "\n".join(lines[-10:]) + "\n"
    stop_server(settings)
    die(f"server failed to start; see {gw_log} / {settings.log_file}\n{tail}")


def spawn_gateway(settings: Settings, model_name: str) -> subprocess.Popen:
    """Launch the gateway as a detached daemon and remember its handle."""
    settings.run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CAPYBARA_MODEL"] = model_name
    log = open(settings.run_dir / "gateway.log", "ab")
    popen_kwargs: Dict[str, Any] = {"stdout": log, "stderr": log,
                                    "stdin": subprocess.DEVNULL}
    if IS_WINDOWS:
        # No setsid on Windows; detach + new group keeps the daemon alive
        # after the CLI returns and avoids console windows popping up.
        popen_kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                         | subprocess.DETACHED_PROCESS)
    else:
        popen_kwargs["start_new_session"] = True
    cmd = [sys.executable, str(settings.server_py), "gateway", "--model", model_name]
    extra = getattr(settings, "engine_args", None) or []
    if extra:
        cmd.append("--")
        cmd.extend(extra)
    proc = subprocess.Popen(cmd, **popen_kwargs)
    (settings.run_dir / "server.pid").write_text(str(proc.pid))
    return proc


def start_server(settings: Settings, model: Path, foreground: bool = False) -> None:
    """Start the full stack (gateway + engine) serving `model`."""
    if not settings.server_bin.exists():
        die(f"engine not found at {settings.server_bin} - place llama-server "
            "there, in this app's bin/ folder, or on PATH")
    foreign_port_guard(settings)
    if port_has_listener(settings, settings.engine_port):
        die(f"internal engine port {settings.engine_port} is already in use - "
            f"free it or pick another public port (CAPYBARA_PORT)")

    if foreground:
        print(f"Capybara {VERSION} starting in foreground - press Ctrl-C to stop")
        sys.stdout.flush()
        env = os.environ.copy()
        env["CAPYBARA_MODEL"] = model.name
        cmd = [sys.executable, str(settings.server_py),
               "gateway", "--model", model.name]
        extra = getattr(settings, "engine_args", None) or []
        if extra:
            cmd.append("--")
            cmd.extend(extra)
        try:
            if IS_WINDOWS:
                # exec* does not replace processes on Windows; run attached.
                proc = subprocess.Popen(cmd, env=env)
                proc.wait()
            else:
                # Replace this process so signals (Ctrl-C, TERM) reach the
                # supervisor itself, which owns the engine child cleanup.
                os.execve(sys.executable,
                          [sys.executable] + cmd,
                          env)
        except KeyboardInterrupt:
            pass
        finally:
            settings.state_file.unlink(missing_ok=True)
        return

    proc = spawn_gateway(settings, model.name)
    wait_until_ready(settings, proc)


def hot_swap(settings: Settings, model: Path) -> bool:
    """Ask a running gateway to load another model; True on success."""
    payload = json.dumps({"model": model.name}).encode("utf-8")
    req = urllib.request.Request(
        f"{settings.base_url}/api/use", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return bool(data.get("ok"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return False


def ensure_server(settings: Settings, model: Path) -> None:
    """Make sure the public endpoint serves exactly this model."""
    state = server_state(settings)
    loaded = state.get("model") if state else None
    if server_ready(settings) and state_is_ours(settings, state):
        if loaded == model.name:
            return
        print(f"switching model: {loaded} -> {model.name}")
        if hot_swap(settings, model):
            return
        # swap failed mid-flight: fall through to a full restart
        stop_server(settings)
    start_server(settings, model)


def trim_history(history: List[Dict[str, str]], max_chars: int) -> List[Dict[str, str]]:
    """Drop the oldest turns so the prompt stays inside the context window.

    System messages are always preserved; everything else is removed oldest
    first until the serialized conversation fits within max_chars.
    """
    def size(msgs: List[Dict[str, str]]) -> int:
        return sum(len(m.get("content", "")) for m in msgs) + 4 * len(msgs)

    if size(history) <= max_chars:
        return history
    system = [m for m in history if m.get("role") == "system"]
    rest = [m for m in history if m.get("role") != "system"]
    while rest and size(system + rest) > max_chars:
        rest = rest[1:]
    # avoid ending on an orphan user message pair boundary issues: fine as-is
    return system + rest


def stream_chat_completions(settings: Settings,
                            payload: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
    """POST to /v1/chat/completions streaming deltas to stdout.

    Returns the full text plus timing statistics (tokens, seconds, tok/s).
    """
    req = urllib.request.Request(
        settings.openai_url + "/chat/completions",
        data=json.dumps({**payload, "stream": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks: List[str] = []
    tokens = 0
    started = time.time()
    first_token: Optional[float] = None
    try:
        resp = urllib.request.urlopen(req, timeout=None)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace").strip()
        raise RuntimeError(f"HTTP {exc.code}: {body[:400]}") from exc
    with resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
                delta = obj["choices"][0]["delta"].get("content") or ""
            except (KeyError, IndexError, ValueError):
                continue
            if delta:
                if first_token is None:
                    first_token = time.time() - started
                chunks.append(delta)
                tokens += 1
                print(delta, end="", flush=True)
    elapsed = time.time() - started
    text = "".join(chunks)
    tps = tokens / elapsed if elapsed > 0 else 0.0
    if tokens:
        dim, reset = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")
        ttft = f", first token {first_token:.1f}s" if first_token else ""
        print(f"{dim}\n[{tokens} tokens · {elapsed:.1f}s{ttft} · {tps:.1f} tok/s]{reset}",
              flush=True)
    return text, {"tokens": float(tokens), "seconds": elapsed, "tok_per_sec": tps}


def chat_payload(model_name: str, history: List[Dict[str, str]],
                 params: Dict[str, Any]) -> Dict[str, Any]:
    """Build the request body for the OpenAI-compatible endpoint."""
    return {"model": model_name, "messages": history, **params}


def read_multiline(first: str) -> str:
    """Collect a triple-quoted multi-line prompt (Ollama REPL style)."""
    body = first[3:]
    end = body.find('"""')
    if end != -1:
        return body[:end]
    parts = [body]
    while True:
        try:
            line = input()
        except EOFError:
            break
        end = line.find('"""')
        if end != -1:
            parts.append(line[:end])
            break
        parts.append(line)
    return "\n".join(parts)


def interactive_chat(settings: Settings, model: Path) -> None:
    """Run a REPL chat session against the given model."""
    system_msgs, params = request_context(model)
    history: List[Dict[str, str]] = list(system_msgs)
    budget = max(1024, int(settings.context * 3.5))
    print(f"Capybara running {model.name}. Type /? for help.")
    while True:
        try:
            raw = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if raw.strip().startswith('"""'):
            prompt = read_multiline(raw.strip())
            if not prompt.strip():
                continue
            history.append({"role": "user", "content": prompt})
            history = trim_history(history, budget)
            try:
                reply, _stats = stream_chat_completions(
                    settings, chat_payload(model.name, history, params))
                history.append({"role": "assistant", "content": reply})
            except Exception as exc:
                history.pop()
                print(f"capybara: {exc}")
            continue
        prompt = raw.strip()
        if prompt in ("/bye", "/exit", "/quit"):
            return
        if prompt == "/clear":
            history = list(system_msgs)
            print("context cleared")
            continue
        if prompt in ("/?", "/help"):
            print("/bye|/exit|/quit  leave the session")
            print("/clear            reset the conversation")
            print("/load MODEL       switch to another installed model")
            print("/show info|system show model details or its system prompt")
            print('/set system TEXT  replace the system prompt')
            print('"""..."""         enter a multi-line message')
            continue
        if prompt.startswith("/load"):
            wanted = prompt[5:].strip()
            target = resolve_model_or_alias(settings, wanted) if wanted else None
            if target is None:
                installed = ", ".join(n for n, _ in group_models(list_models(settings)))
                print(f"capybara: unknown model '{wanted}'. Installed: {installed}")
                continue
            model = target
            system_msgs, params = request_context(model)
            history = list(system_msgs)
            print(f"switched to {model.name}")
            continue
        if prompt.startswith("/show"):
            what = prompt[5:].strip() or "info"
            meta = sidecar_for(model) or {}
            if what == "system":
                print(meta.get("system") or "(no system prompt)")
            else:
                print(f"model    {model.name}")
                print(f"size     {human_size(model.stat().st_size)}")
                if meta.get("base"):
                    print(f"base     {meta['base']}")
                pretty = ", ".join(f"{k}={v}" for k, v in sorted((meta.get("params") or {}).items()))
                if pretty:
                    print(f"params   {pretty}")
            continue
        if prompt.startswith("/set system"):
            text = prompt[len("/set system"):].strip().strip('"').strip("'")
            if text:
                system_msgs = [{"role": "system", "content": text}]
            else:
                system_msgs = []
            history = list(system_msgs)
            print("system prompt updated" if text else "system prompt cleared")
            continue
        if not prompt:
            continue
        history.append({"role": "user", "content": prompt})
        history = trim_history(history, budget)
        try:
            reply, _stats = stream_chat_completions(
                settings, chat_payload(model.name, history, params))
            history.append({"role": "assistant", "content": reply})
        except SystemExit:
            raise
        except RuntimeError as exc:
            history.pop()  # drop the failed turn so the loop can continue
            print(f"capybara: {exc}")
        except Exception as exc:
            history.pop()
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


def do_logs(settings: Settings, lines: int) -> None:
    """Handle the 'logs' command."""
    if not settings.log_file.exists():
        print("No logs yet.")
        return
    content = settings.log_file.read_text(errors="replace").splitlines()
    print("\n".join(content[-lines:]))


def _scroll_ticker(text: str, width: int, duration: float = 2.0) -> None:
    """Scroll text horizontally in a terminal line."""
    if not text:
        return
    padded = text + " " * width
    steps = min(len(padded), int(duration / 0.02))
    for i in range(steps):
        visible = padded[i:i + width]
        sys.stdout.write(f"\033[2K\033[1G\033[2m{visible}\033[0m")
        sys.stdout.flush()
        time.sleep(0.02)
    sys.stdout.write("\033[2K\033[1G")
    sys.stdout.flush()


def show_list_with_ticker(items: List[str], title: str) -> int:
    """Show a list, scrolling the last line if results overflow the terminal."""
    try:
        term_height = shutil.get_terminal_size().lines
    except Exception:
        term_height = 24
    term_height = max(term_height, 5)

    visible = term_height - 3
    if visible < 1:
        visible = 1

    print(f"\n{title} ({len(items)} results):")

    if len(items) <= visible:
        for i, item in enumerate(items, 1):
            print(f"{i}. {item}")
        return int(input("Select number: ").strip()) - 1

    shown = items[:visible]
    remaining = items[visible:]

    for i, item in enumerate(shown, 1):
        print(f"{i}. {item}")

    if remaining:
        ticker = "  |  ".join(
            f"[{i + visible}] {item}" for i, item in enumerate(remaining)
        )
        try:
            _scroll_ticker(ticker, term_height)
        except Exception:
            pass

    while True:
        try:
            choice = int(input("Select number: ").strip()) - 1
            if 0 <= choice < len(items):
                return choice
            print(f"Enter a number between 1 and {len(items)}")
        except ValueError:
            print("Enter a number")


def do_search(settings: Settings, query: str) -> None:
    """Search Hugging Face for GGUF models."""
    search_hf(settings, query)


def search_hf(settings: Settings, query: str) -> None:
    """Search Hugging Face, show repos with GGUF files, then pick a file."""
    url = (
        "https://huggingface.co/api/models"
        f"?search={urllib.parse.quote(query)}&full=true&limit=50"
    )
    try:
        results = http_json(url)
    except Exception as exc:
        die(f"Hugging Face search failed: {exc}")

    repos = []
    for r in results:
        if "id" not in r:
            continue
        siblings = [s.get("rfilename", "") for s in r.get("siblings", [])]
        if any(name.lower().endswith(".gguf") for name in siblings):
            repos.append(r["id"])

    if not repos:
        print("No GGUF models found.")
        return

    idx = show_list_with_ticker(repos, "Hugging Face GGUF models")
    repo = repos[idx]

    ggufs = hf_repo_ggufs(repo)
    if not ggufs:
        die(f"No GGUF files in {repo}")

    print(f"\nGGUF files in {repo}:")
    for i, g in enumerate(ggufs, 1):
        print(f"{i}. {g}")

    file_idx = 0
    if len(ggufs) > 1:
        file_idx = int(input(f"Select file (1-{len(ggufs)}): ").strip()) - 1
    if file_idx < 0 or file_idx >= len(ggufs):
        return

    filename = ggufs[file_idx]
    dest = settings.models / filename
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already installed: {dest}")
        return

    download_url = (
        f"https://huggingface.co/{repo}/resolve/main/"
        f"{urllib.parse.quote(filename)}"
    )
    download_to(dest, download_url)
    print(f"Installed {dest}")


def do_serve(settings: Settings, args: argparse.Namespace) -> None:
    """Handle the 'serve' command."""
    spec = args.model or os.environ.get("CAPYBARA_MODEL")
    model = resolve_model(settings, spec) if spec else None
    if model is None:
        installed = list_models(settings)
        if not installed:
            die("no models installed - try: capybara pull smollm")
        model = installed[0]
    state = server_state(settings)
    already_up = server_ready(settings) and state_is_ours(settings, state)
    if already_up and not args.foreground:
        if state.get("model") != model.name:
            print(f"switching model: {state.get('model')} -> {model.name}")
            hot_swap(settings, model)
    else:
        start_server(settings, model, foreground=args.foreground)
    if not args.foreground:
        print(f"Capybara serving {model.name}")
        print(f"Web UI:     http://{settings.host}:{settings.port}/")
        print(f"OpenAI API: {settings.openai_url}/chat/completions")


def do_ps(settings: Settings) -> None:
    """Handle the 'ps' command."""
    state = server_state(settings)
    if not state or not server_ready(settings) or not state_is_ours(settings, state):
        print("STOPPED")
        return
    uptime = time.time() - float(state.get("started_at", time.time()))
    if state.get("idle_unloaded") or not state.get("model"):
        print(f"GATEWAY  url=http://{state.get('host')}:{state.get('port')}  "
              f"uptime={int(uptime)}s  model=(unloaded, idle)")
        return
    print(f"RUNNING  mode={state.get('mode', 'gateway')}  "
          f"model={state.get('model')}  "
          f"url=http://{state.get('host')}:{state.get('port')}  "
          f"engine_pid={state.get('pid')}  uptime={int(uptime)}s")


def open_ui(settings: Settings, model: Optional[Path] = None) -> None:
    """Make sure the server is up, then open the Open WebUI frontend.

    Launch order: reuse an Open WebUI already listening on webui_port,
    else start one via the ``open-webui`` command if installed,
    else print setup instructions.
    """
    if model is None:
        model = resolve_model(settings, os.environ.get("CAPYBARA_MODEL", ""))
    if model is None:
        installed = list_models(settings)
        if not installed:
            die("no models installed - try: capybara pull smollm")
        model = installed[0]
    state = server_state(settings)
    if not (server_ready(settings) and state_is_ours(settings, state)):
        start_server(settings, model)

    api_url = f"http://{settings.host}:{settings.port}/v1"
    ui_url = f"http://{settings.host}:{settings.webui_port}/"

    def webui_healthy() -> bool:
        try:
            with urllib.request.urlopen(f"{ui_url}health", timeout=1) as resp:
                return resp.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            return False

    if port_has_listener(settings, settings.webui_port):
        print(f"Open WebUI already running: {ui_url}")
        webbrowser.open(ui_url)
        return

    launcher = shutil.which("open-webui")
    if not launcher:
        die("Open WebUI is not installed. Install it with either:\n"
            "  pip install open-webui\n"
            "or:\n"
            "  docker run -d --name capybara-webui -p 8080:8080 \\\n"
            f"    -e OPENAI_API_BASE_URL={api_url.replace('127.0.0.1', 'host.docker.internal')} \\\n"
            "    -e OPENAI_API_KEY=capybara -e WEBUI_AUTH=false \\\n"
            "    --add-host=host.docker.internal:host-gateway \\\n"
            "    -v capybara-webui:/app/backend/data --restart unless-stopped \\\n"
            "    ghcr.io/open-webui/open-webui:main\n"
            "then run 'capybara ui' again.")

    data_dir = settings.home / "webui"
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.home / "logs"
    log_path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "DATA_DIR": str(data_dir),
        "OPENAI_API_BASE_URL": api_url,
        "OPENAI_API_KEYS": "capybara",
        "WEBUI_AUTH": "false",
    })
    with open(log_path / "webui.log", "ab") as log:
        popen_kwargs: Dict[str, Any] = {"stdout": log, "stderr": log}
        if IS_WINDOWS:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen([launcher, "serve", "--port", str(settings.webui_port)],
                                **popen_kwargs)
    print(f"starting Open WebUI (pid {proc.pid}) - first boot can take a minute...")
    deadline = time.time() + 180
    while time.time() < deadline:
        if webui_healthy():
            break
        if proc.poll() is not None:
            die(f"Open WebUI exited with code {proc.returncode} - "
                f"see {log_path / 'webui.log'}")
        time.sleep(1)
    else:
        die("Open WebUI did not become ready within 180s - "
            f"see {log_path / 'webui.log'}")
    print(f"Capybara API: {api_url}")
    print(f"Open WebUI:   {ui_url}")
    webbrowser.open(ui_url)


# --- coding agents -----------------------------------------------------




def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="capybara",
        description="Run GGUF models locally via llama.cpp. Single-file CLI + gateway.",
        add_help=False,
    )
    parser.add_argument("--help", action="help", help="Show this help message")
    sub = parser.add_subparsers(dest="cmd")

    serve = sub.add_parser("serve", help="start web UI + API server")
    serve.add_argument("--model", help="model to load")
    serve.add_argument("-F", "--foreground", action="store_true",
                       help="run in foreground (no daemon)")

    sub.add_parser("ui", help="open browser UI")

    run = sub.add_parser("run", help="chat or one-shot prompt")
    run.add_argument("model")
    run.add_argument("prompt", nargs="?")

    pull = sub.add_parser("pull", help="download a model")
    pull.add_argument("model")

    sub.add_parser("list", aliases=["ls"], help="list installed models")

    show = sub.add_parser("show", help="model details")
    show.add_argument("model")

    rm = sub.add_parser("rm", help="remove a model")
    rm.add_argument("model")

    cp = sub.add_parser("cp", help="copy model under new name")
    cp.add_argument("source")
    cp.add_argument("destination")

    create = sub.add_parser("create", help="create model from Modelfile")
    create.add_argument("-f", "--file", default="Modelfile")
    create.add_argument("name")

    sub.add_parser("ps", help="server status")
    sub.add_parser("stop", help="stop server")

    logs = sub.add_parser("logs", help="engine logs")
    logs.add_argument("-n", type=int, default=50, help="lines to show (default 50)")

    search = sub.add_parser("search", help="search GGUF models on HF")
    search.add_argument("query", help="search query")

    sub.add_parser("help", help="show help")

    sub.add_parser("gateway", help=argparse.SUPPRESS)

    return parser





def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)

    engine_args: List[str] = []
    if "--" in argv:
        idx = argv.index("--")
        engine_args = argv[idx + 1:]
        argv = argv[:idx]

    print(
        "           ...:...                                .=*+.\n"
        "        ..=*******=.                              .=*+.\n"
        "       .-***-. ..-+.                              .=*+.\n"
        "       .+*+.      ...:=++++-. .=+--+*+-..=+=.  :+=.=*+-=++=:...-++++=:. -+=-++:.-++++=:.\n"
        "       :**=          .-:.:=*+..+**-::+**::**-..+*:.=**=::=**- .-:..-*+: -**=-:..::..-**:.\n"
        "       .+**:.     ...-+**++*+..+*-.  .+*=.-*+:+*-..=*+.  .=*+.:+**++*+: -*+.   :+**++**-.\n"
        "       .:+**=:::-=+.-*+:..=*+..+*+:..-**-. -***=. .=**-..:+*=.+*-..-*+: -*+.  .+*- .-**-.\n"
        "        ..-+******-..+***++*+..+*++***+:.  .=*+.  .=*+=****-..-***+=**: -*+.  .-***++**-.\n"
        "             ....      ..     .+*-....    .-*+.         .       ..               ..\n"
        "                              .+*-.       :**:\n"
        "                              .....      .....\n"
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()
    settings.engine_args = engine_args  # type: ignore[attr-defined]
    if args.cmd == "serve":
        do_serve(settings, args)
    elif args.cmd == "ui":
        open_ui(settings)
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
    elif args.cmd == "search":
        do_search(settings, args.query)
    elif args.cmd == "gateway":
        run_gateway()
        return
    elif args.cmd in (None, "help"):
        parser.print_help()
        return
    else:
        die(f"unknown command: {args.cmd}")



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
    match = SHARD_RE.search(name)
    stem = (match.group(1) if match else Path(name).stem).lower()
    hit = next((m for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([bm])(?![a-z0-9])",
                                       stem)), None)
    return f"{hit.group(1)}{hit.group(2).upper()}" if hit else ""


def quant_guess(name: str) -> str:
    """Extract a quantisation label like 'Q4_K_M' from a model name."""
    stem = Path(name).stem.upper()
    hit = QUANT_SUFFIX_RE.search(stem)
    return hit.group(0).lstrip("-._") if hit else ""


class EngineManager:
    """Spawn, monitor, hot-swap and idle-unload the llama-server child."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.proc: Optional[subprocess.Popen] = None
        self.model: Optional[Path] = None
        self.started_at: float = 0.0
        self.swapping = False
        self.unloaded_idle = False
        self.keep_alive = parse_keep_alive(settings.keep_alive_raw, default=300.0)
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
            self.keep_alive = parse_keep_alive(keep_alive, default=self.keep_alive)

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
        meta = sidecar_for(model) or {}
        context = int((meta.get("params") or {}).get("num_ctx", self.settings.context))
        cmd = [
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
        cmd.extend(getattr(self, "engine_args", []))
        return cmd

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
                "version": VERSION,
                "api": "Ollama-compatible (/api/chat, /api/generate, /api/tags)"
                       " + OpenAI-compatible (/v1/chat/completions)",
                "ui": "run 'capybara ui' for the Open WebUI frontend",
            })
        elif path == "/api/version":
            self._send_json(200, {"version": VERSION})
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
                   b'<text y="80" font-size="80">\\U0001F42D</text></svg>')
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
            "version": VERSION,
        }

    def models(self) -> Dict[str, Any]:
        settings = self.manager.settings
        loaded_key = None
        if self.manager.model is not None:
            match = SHARD_RE.search(self.manager.model.name)
            stem = self.manager.model.stem
            loaded_key = match.group(1) if match else stem
        entries = []
        for name, size in group_models(list_models(settings)):
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
        model = resolve_model_or_alias(settings, str(wanted))
        if model is None:
            installed = ", ".join(name for name, _ in group_models(
                list_models(settings))) or "(none)"
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
        for path in list_models(settings):
            match = SHARD_RE.search(path.name)
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
        model = resolve_model_or_alias(mgr.settings, name) if name else mgr.model
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
        sidecar_msgs, sidecar_params = request_context(model)
        has_system = any(m["role"] == "system" for m in msgs)
        if sidecar_msgs and not has_system:
            msgs = sidecar_msgs + msgs
        payload: Dict[str, Any] = {"model": model.name, "messages": msgs,
                                   **sidecar_params, "stream": True,
                                   "stream_options": {"include_usage": True}}
        payload.update(ollama_options_to_openai(body.get("options")))
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
            payload.update(ollama_options_to_openai(body.get("options")))
        else:
            sidecar_msgs, sidecar_params = request_context(model)
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
            payload.update(ollama_options_to_openai(body.get("options")))
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
        model = resolve_model_or_alias(settings, str(body.get("model") or ""))
        if model is None:
            self.send_error_reply(404, f"model '{body.get('model')}' not found")
            return
        meta = sidecar_for(model) or {}
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
                pull_model(settings, spec)
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
        paths = [p for p in list_models(settings)
                 if (SHARD_RE.search(p.name).group(1) if SHARD_RE.search(p.name)
                     else p.stem) == wanted]
        if not paths:
            self.send_error_reply(404, f"model '{wanted}' not found")
            return
        loaded_key = None
        if self.manager.model is not None:
            match = SHARD_RE.search(self.manager.model.name)
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
    system_msgs, params = request_context(mgr.model)
    messages = body.get("messages") or []
    has_system = any(m.get("role") == "system" for m in messages)
    if system_msgs and not has_system:
        body["messages"] = system_msgs + messages
    for key, value in params.items():
        body.setdefault(key, value)
    return body


def run_gateway(argv: Optional[list] = None) -> int:
    """Gateway entry point: start engine, then serve forever."""
    args = list(sys.argv[1:] if argv is None else argv)
    engine_args: List[str] = []
    if "--" in args:
        idx = args.index("--")
        engine_args = args[idx + 1:]
        args = args[:idx]
    wanted = args[args.index("--model") + 1] if "--model" in args else None
    settings = load_settings()
    mgr = EngineManager(settings)
    mgr.engine_args = engine_args

    model = None
    if wanted:
        model = resolve_model(settings, wanted)
    if model is None:
        installed = sorted(list_models(settings),
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
    print(f"Capybara {VERSION} serving {mgr.model.name} "
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
    main()

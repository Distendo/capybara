#!/usr/bin/env bash

# Capybara ez edition lol
# Dont install and think like this is better version of Ollama. This was made for placeholder.

set -euo pipefail

CAPYBARA_HOME="${CAPYBARA_HOME:-$HOME/.capybara}"
CAPYBARA_BIN="$CAPYBARA_HOME/bin"
CAPYBARA_MODELS="$CAPYBARA_HOME/models"
CAPYBARA_SOURCE="$CAPYBARA_HOME/llama.cpp"
CAPYBARA_RUN="$CAPYBARA_HOME/run"
CAPYBARA_CONFIG="$CAPYBARA_HOME/config"

# Prefer ~/.local/bin because it doesn't require sudo.
INSTALL_BIN="${CAPYBARA_INSTALL_BIN:-$HOME/.local/bin}"

SERVER="$CAPYBARA_BIN/llama-server"

log() {
    printf '\033[1;36mcapybara\033[0m %s\n' "$1"
}

warn() {
    printf '\033[1;33mcapybara\033[0m %s\n' "$1"
}

die() {
    printf '\033[1;31mcapybara\033[0m %s\n' "$1" >&2
    exit 1
}

has() {
    command -v "$1" >/dev/null 2>&1
}

cores() {
    if has nproc; then
        nproc
    elif has sysctl; then
        sysctl -n hw.ncpu
    else
        echo 4
    fi
}

detect_os() {
    UNAME="$(uname -s)"
    ARCH="$(uname -m)"

    case "$UNAME" in
        Darwin)
            OS="macos"
            ;;

        Linux)
            OS="linux"
            ;;

        FreeBSD)
            OS="freebsd"
            ;;

        *)
            die "Unsupported operating system: $UNAME"
            ;;
    esac

    log "Detected $OS / $ARCH"
}

linux_package_manager() {
    if has apt-get; then
        echo apt
    elif has dnf; then
        echo dnf
    elif has yum; then
        echo yum
    elif has pacman; then
        echo pacman
    elif has zypper; then
        echo zypper
    else
        echo unknown
    fi
}

install_dependencies() {
    log "Installing dependencies"

    case "$OS" in
        macos)
            has brew || die "Homebrew is required on macOS: https://brew.sh"

            brew install \
                git \
                cmake \
                ninja \
                pkg-config \
                python3 \
                curl \
                jq \
                2>/dev/null || true
            ;;

        linux)
            PM="$(linux_package_manager)"

            case "$PM" in
                apt)
                    sudo apt-get update

                    sudo apt-get install -y \
                        git \
                        cmake \
                        ninja-build \
                        build-essential \
                        pkg-config \
                        python3 \
                        python3-venv \
                        curl \
                        jq
                    ;;

                dnf)
                    sudo dnf install -y \
                        git \
                        cmake \
                        ninja-build \
                        gcc \
                        gcc-c++ \
                        make \
                        pkgconf-pkg-config \
                        python3 \
                        curl \
                        jq
                    ;;

                yum)
                    sudo yum install -y \
                        git \
                        cmake \
                        ninja-build \
                        gcc \
                        gcc-c++ \
                        make \
                        pkgconfig \
                        python3 \
                        curl \
                        jq
                    ;;

                pacman)
                    sudo pacman -Sy --needed --noconfirm \
                        git \
                        cmake \
                        ninja \
                        base-devel \
                        pkgconf \
                        python \
                        curl \
                        jq
                    ;;

                zypper)
                    sudo zypper --non-interactive install \
                        git \
                        cmake \
                        ninja \
                        gcc \
                        gcc-c++ \
                        make \
                        pkg-config \
                        python3 \
                        curl \
                        jq
                    ;;

                *)
                    die "Couldn't identify a Linux package manager"
                    ;;
            esac
            ;;

        freebsd)
            sudo pkg update

            sudo pkg install -y \
                git \
                cmake \
                ninja \
                gcc \
                pkgconf \
                python3 \
                curl \
                jq
            ;;
    esac
}

detect_gpu() {
    BACKEND="cpu"
    GPU_NAME="CPU"

    # Apple Silicon
    if [[ "$OS" == "macos" && "$ARCH" == "arm64" ]]; then
        BACKEND="metal"
        GPU_NAME="Apple Metal"
        return
    fi

    # NVIDIA
    if has nvidia-smi; then
        BACKEND="cuda"

        GPU_NAME="$(
            nvidia-smi \
                --query-gpu=name \
                --format=csv,noheader \
                2>/dev/null |
                head -n 1
        )"

        GPU_NAME="${GPU_NAME:-NVIDIA CUDA}"
        return
    fi

    # AMD
    if has rocminfo || has hipconfig; then
        BACKEND="rocm"

        if has rocminfo; then
            GPU_NAME="AMD ROCm"
        else
            GPU_NAME="AMD HIP"
        fi

        return
    fi

    # Intel
    if has sycl-ls; then
        BACKEND="sycl"
        GPU_NAME="Intel SYCL"
        return
    fi

    # Vulkan fallback
    if has vulkaninfo; then
        BACKEND="vulkan"
        GPU_NAME="Vulkan GPU"
        return
    fi

    log "No supported GPU backend found"
    log "Using optimized CPU backend"
}

clone_engine() {
    if [[ -d "$CAPYBARA_SOURCE/.git" ]]; then
        log "Updating llama.cpp"

        git -C "$CAPYBARA_SOURCE" fetch \
            --depth=1 \
            origin master >/dev/null 2>&1 || true

        git -C "$CAPYBARA_SOURCE" reset \
            --hard \
            origin/master >/dev/null 2>&1 || true
    else
        log "Downloading llama.cpp"

        rm -rf "$CAPYBARA_SOURCE"

        git clone \
            --depth=1 \
            https://github.com/ggml-org/llama.cpp.git \
            "$CAPYBARA_SOURCE"
    fi
}

build_engine() {
    BUILD="$CAPYBARA_SOURCE/build-$BACKEND"

    rm -rf "$BUILD"

    log "Building engine: $BACKEND"

    CMAKE_ARGS=(
        -S "$CAPYBARA_SOURCE"
        -B "$BUILD"
        -G Ninja
        -DCMAKE_BUILD_TYPE=Release
        -DGGML_NATIVE=ON
        -DLLAMA_BUILD_SERVER=ON
    )

    case "$BACKEND" in
        metal)
            CMAKE_ARGS+=(
                -DGGML_METAL=ON
                -DGGML_METAL_EMBED_LIBRARY=ON
            )
            ;;

        cuda)
            CMAKE_ARGS+=(
                -DGGML_CUDA=ON
                -DGGML_CUDA_FA_ALL_QUANTS=ON
                -DGGML_CUDA_ENABLE_UNIFIED_MEMORY=ON
            )
            ;;

        rocm)
            CMAKE_ARGS+=(
                -DGGML_HIP=ON
            )
            ;;

        sycl)
            CMAKE_ARGS+=(
                -DGGML_SYCL=ON
            )
            ;;

        vulkan)
            CMAKE_ARGS+=(
                -DGGML_VULKAN=ON
            )
            ;;

        cpu)
            CMAKE_ARGS+=(
                -DGGML_OPENMP=ON
                -DGGML_CPU_ALL_VARIANTS=ON
            )
            ;;
    esac

    cmake "${CMAKE_ARGS[@]}"

    cmake \
        --build "$BUILD" \
        --config Release \
        --parallel "$(cores)"

    mkdir -p "$CAPYBARA_BIN"

    FOUND_SERVER=""

    for FILE in \
        "$BUILD/bin/llama-server" \
        "$BUILD/bin/server" \
        "$BUILD/bin/llama-server.exe"
    do
        if [[ -f "$FILE" ]]; then
            FOUND_SERVER="$FILE"
            break
        fi
    done

    [[ -n "$FOUND_SERVER" ]] ||
        die "llama-server compilation failed"

    cp "$FOUND_SERVER" "$SERVER"
    chmod +x "$SERVER"

    if [[ -f "$BUILD/bin/llama" ]]; then
        cp "$BUILD/bin/llama" "$CAPYBARA_BIN/llama"
        chmod +x "$CAPYBARA_BIN/llama"
    fi
}

create_config() {
    mkdir -p \
        "$CAPYBARA_HOME" \
        "$CAPYBARA_BIN" \
        "$CAPYBARA_MODELS" \
        "$CAPYBARA_RUN"

    cat > "$CAPYBARA_CONFIG" <<EOF
CAPYBARA_HOME=$CAPYBARA_HOME
CAPYBARA_MODELS=$CAPYBARA_MODELS
CAPYBARA_BACKEND=$BACKEND
CAPYBARA_GPU=$GPU_NAME
CAPYBARA_HOST=127.0.0.1
CAPYBARA_PORT=11434
EOF
}

create_cli() {
    log "Installing CLI"

    cat > "$CAPYBARA_BIN/capybara" <<'PY'
#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


HOME = Path(os.environ.get(
    "CAPYBARA_HOME",
    str(Path.home() / ".capybara")
))

BIN = HOME / "bin"
MODELS = HOME / "models"
RUN = HOME / "run"
CONFIG = HOME / "config"

SERVER = BIN / "llama-server"

HOST = os.environ.get("CAPYBARA_HOST", "127.0.0.1")
PORT = int(os.environ.get("CAPYBARA_PORT", "11434"))

BASE = f"http://{HOST}:{PORT}"
OPENAI = f"{BASE}/v1"


def die(message):
    print(f"capybara: {message}", file=sys.stderr)
    sys.exit(1)


def request(path, data=None, method="GET", timeout=10):
    url = BASE + path

    body = None

    headers = {
        "Content-Type": "application/json"
    }

    if data is not None:
        body = json.dumps(data).encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()

            if not raw:
                return {}

            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw.decode(errors="replace")

    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        die(f"HTTP {e.code}: {raw}")

    except urllib.error.URLError:
        return None


def model_path(name):
    safe = name.replace("/", "_").replace(":", "_")

    candidate = MODELS / f"{safe}.gguf"

    if candidate.exists():
        return candidate

    matches = list(MODELS.glob(f"{safe}*.gguf"))

    if matches:
        return matches[0]

    direct = Path(name)

    if direct.exists() and direct.is_file():
        return direct

    return candidate


def all_models():
    MODELS.mkdir(parents=True, exist_ok=True)

    return sorted(
        MODELS.glob("*.gguf"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )


def api_running():
    result = request("/health", timeout=2)
    return result is not None


def command_serve(args):
    if not SERVER.exists():
        die("llama-server is not installed")

    if api_running():
        print("Capybara is already running")
        return

    log = RUN / "server.log"
    pid_file = RUN / "server.pid"

    RUN.mkdir(parents=True, exist_ok=True)

    model = os.environ.get("CAPYBARA_MODEL")

    if not model:
        models = all_models()

        if models:
            model = str(models[0])

    if not model:
        die(
            "no model found. "
            "Put a .gguf file in ~/.capybara/models "
            "or set CAPYBARA_MODEL"
        )

    threads = os.environ.get("CAPYBARA_THREADS")

    if not threads:
        threads = str(os.cpu_count() or 4)

    context = os.environ.get("CAPYBARA_CONTEXT", "8192")
    batch = os.environ.get("CAPYBARA_BATCH", "2048")
    gpu_layers = os.environ.get("CAPYBARA_GPU_LAYERS", "999")
    parallel = os.environ.get("CAPYBARA_PARALLEL", "1")

    cmd = [
        str(SERVER),

        "--model",
        model,

        "--host",
        HOST,

        "--port",
        str(PORT),

        "--threads",
        threads,

        "--threads-batch",
        threads,

        "--ctx-size",
        context,

        "--batch-size",
        batch,

        "--ubatch-size",
        batch,

        "--n-gpu-layers",
        gpu_layers,

        "--parallel",
        parallel,

        "--cont-batching",

        "--flash-attn",
        "on",
    ]

    if args.extra:
        cmd.extend(args.extra)

    with open(log, "ab") as output:
        process = subprocess.Popen(
            cmd,
            stdout=output,
            stderr=output
        )

    pid_file.write_text(str(process.pid))

    time.sleep(1)

    print("Capybara server started")
    print(f"API: http://{HOST}:{PORT}")


def command_stop(args):
    target = args.model

    pid_file = RUN / "server.pid"

    if not pid_file.exists():
        print("No running Capybara server")
        return

    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, signal.SIGTERM)

        try:
            pid_file.unlink()
        except FileNotFoundError:
            pass

        print("Capybara stopped")

    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        print("Capybara was not running")


def choose_model(name=None):
    if name:
        path = model_path(name)

        if not path.exists():
            die(f"model not found: {name}")

        return path

    models = all_models()

    if not models:
        die("no models installed")

    return models[0]


def command_run(args):
    model = choose_model(args.model)

    if not api_running():
        os.environ["CAPYBARA_MODEL"] = str(model)

        subprocess.run(
            [sys.argv[0], "serve"],
            check=True
        )

        for _ in range(30):
            if api_running():
                break
            time.sleep(0.25)

    if args.prompt:
        chat(model.name, args.prompt)
        return

    print(f"Capybara running {model.name}")
    print('Type /bye to exit')

    history = []

    while True:
        try:
            prompt = input(">>> ")

        except (EOFError, KeyboardInterrupt):
            print()
            break

        if prompt.strip() in {"/bye", "/exit", "/quit"}:
            break

        if not prompt.strip():
            continue

        messages = history + [
            {
                "role": "user",
                "content": prompt
            }
        ]

        data = {
            "model": model.name,
            "messages": messages,
            "stream": True
        }

        try:
            stream_chat(data)
        except Exception as e:
            print(f"\ncapybara: {e}")

        history.append({
            "role": "user",
            "content": prompt
        })


def chat(model, prompt):
    data = {
        "model": Path(model).name,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": True
    }

    stream_chat(data)


def stream_chat(data):
    body = json.dumps(data).encode()

    req = urllib.request.Request(
        f"{OPENAI}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=None) as response:
        while True:
            line = response.readline()

            if not line:
                break

            line = line.decode(errors="replace").strip()

            if not line.startswith("data:"):
                continue

            chunk = line[5:].strip()

            if chunk == "[DONE]":
                break

            try:
                obj = json.loads(chunk)

                choices = obj.get("choices", [])

                if choices:
                    delta = choices[0].get("delta", {})
                    text = delta.get("content", "")

                    if text:
                        print(text, end="", flush=True)

            except json.JSONDecodeError:
                pass

    print()


def command_list(args):
    models = all_models()

    print("NAME\tSIZE\tMODIFIED")

    for model in models:
        stat = model.stat()

        size = stat.st_size

        if size >= 1024 ** 3:
            readable = f"{size / 1024**3:.2f} GB"
        elif size >= 1024 ** 2:
            readable = f"{size / 1024**2:.1f} MB"
        else:
            readable = f"{size / 1024:.1f} KB"

        modified = time.strftime(
            "%Y-%m-%d %H:%M",
            time.localtime(stat.st_mtime)
        )

        print(
            f"{model.stem}\t"
            f"{readable}\t"
            f"{modified}"
        )


def command_ps(args):
    pid_file = RUN / "server.pid"

    if not pid_file.exists():
        print("NAME\tSTATUS")
        return

    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)

        model = os.environ.get(
            "CAPYBARA_MODEL",
            "loaded model"
        )

        print("NAME\tSTATUS")
        print(f"{Path(model).stem}\tRUNNING")

    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        print("NAME\tSTATUS")


def command_rm(args):
    path = model_path(args.model)

    if not path.exists():
        die(f"model not found: {args.model}")

    path.unlink()

    print(f"deleted {args.model}")


def command_cp(args):
    source = model_path(args.source)

    if not source.exists():
        die(f"model not found: {args.source}")

    destination = model_path(args.destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(source, destination)

    print(
        f"copied {args.source} "
        f"to {args.destination}"
    )


def command_show(args):
    path = model_path(args.model)

    if not path.exists():
        die(f"model not found: {args.model}")

    print(f"Model: {args.model}")
    print(f"File: {path}")
    print(f"Size: {path.stat().st_size} bytes")

    try:
        result = subprocess.run(
            [
                str(BIN / "llama"),
                "-m",
                str(path),
                "--info"
            ],
            capture_output=True,
            text=True
        )

        if result.stdout:
            print(result.stdout)

    except Exception:
        pass


def command_pull(args):
    source = args.model
    destination = model_path(source)

    if source.startswith(("http://", "https://")):
        destination = MODELS / (
            source.rstrip("/").split("/")[-1]
        )

        if not destination.name.lower().endswith(".gguf"):
            destination = destination.with_suffix(".gguf")

        print(f"Downloading {source}")

        urllib.request.urlretrieve(
            source,
            destination
        )

        print(f"Downloaded to {destination}")

        return

    if Path(source).exists():
        source_path = Path(source)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            source_path,
            destination
        )

        print(f"Imported {source}")
        return

    registry = os.environ.get(
        "CAPYBARA_REGISTRY_URL",
        ""
    ).rstrip("/")

    if not registry:
        die(
            f"model '{source}' is not a direct file or URL.\n"
            "Set CAPYBARA_REGISTRY_URL to use a Capybara registry."
        )

    url = f"{registry}/{source}.gguf"

    print(f"Downloading {source}")

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    urllib.request.urlretrieve(
        url,
        destination
    )

    print(f"Installed {source}")


def command_create(args):
    modelfile = Path(args.file)

    if not modelfile.exists():
        die(f"Modelfile not found: {modelfile}")

    from_model = None

    for line in modelfile.read_text().splitlines():
        line = line.strip()

        if line.upper().startswith("FROM "):
            from_model = line[5:].strip()
            break

    if not from_model:
        die("Modelfile needs a FROM line")

    source = model_path(from_model)

    if not source.exists():
        die(
            f"base model '{from_model}' isn't installed"
        )

    destination = model_path(args.name)

    shutil.copy2(
        source,
        destination
    )

    metadata = destination.with_suffix(".json")

    config = {
        "name": args.name,
        "base": from_model,
        "modelfile": str(modelfile),
        "created": time.time()
    }

    metadata.write_text(
        json.dumps(
            config,
            indent=2
        )
    )

    print(f"created {args.name}")


def command_push(args):
    registry = os.environ.get(
        "CAPYBARA_REGISTRY_URL",
        ""
    ).rstrip("/")

    if not registry:
        die(
            "CAPYBARA_REGISTRY_URL is required for push"
        )

    source = model_path(args.model)

    if not source.exists():
        die(f"model not found: {args.model}")

    url = f"{registry}/{args.model}.gguf"

    print(
        "Push requires a registry supporting "
        "authenticated uploads."
    )

    print(f"Target: {url}")

    die(
        "No upload protocol configured for this registry"
    )


def command_generate(args):
    model = choose_model(args.model)

    if not api_running():
        os.environ["CAPYBARA_MODEL"] = str(model)

        subprocess.run(
            [sys.argv[0], "serve"],
            check=True
        )

        time.sleep(1)

    data = {
        "model": model.name,
        "prompt": args.prompt,
        "stream": True
    }

    req = urllib.request.Request(
        f"{BASE}/api/generate",
        data=json.dumps(data).encode(),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=None) as response:
        for line in response:
            try:
                obj = json.loads(line)

                text = obj.get(
                    "response",
                    ""
                )

                if text:
                    print(
                        text,
                        end="",
                        flush=True
                    )

            except json.JSONDecodeError:
                pass

    print()


def command_embeddings(args):
    model = choose_model(args.model)

    data = {
        "model": model.name,
        "input": args.text
    }

    result = request(
        "/v1/embeddings",
        data=data,
        method="POST"
    )

    if result is None:
        die("Capybara is not running")

    print(
        json.dumps(
            result,
            indent=2
        )
    )


def command_signin(args):
    token = args.token

    if not token:
        try:
            token = input("Capybara registry token: ")
        except KeyboardInterrupt:
            print()
            return

    token_file = HOME / "token"

    token_file.write_text(token)
    token_file.chmod(0o600)

    print("signed in")


def command_signout(args):
    token_file = HOME / "token"

    token_file.unlink(missing_ok=True)

    print("signed out")


def command_launch(args):
    target = args.integration

    mapping = {
        "opencode": "opencode",
        "claude": "claude",
        "claude-code": "claude",
        "codex": "codex",
        "droid": "droid",
        "copilot": "copilot",
        "openclaw": "openclaw"
    }

    program = mapping.get(
        target,
        target
    )

    if not shutil.which(program):
        die(
            f"integration '{program}' "
            "isn't installed"
        )

    env = os.environ.copy()

    env["OPENAI_BASE_URL"] = OPENAI
    env["OPENAI_API_BASE"] = OPENAI
    env["CAPYBARA_BASE_URL"] = BASE

    model = args.model

    if model:
        env["CAPYBARA_MODEL"] = model

    subprocess.run(
        [program],
        env=env,
        check=False
    )


def command_version(args):
    print("Capybara 0.1.0")
    print("Engine: llama.cpp")
    print(f"API: {BASE}")


def help_text():
    print(
        """Capybara

Usage:
  capybara [command]

Commands:
  serve       Start Capybara
  create      Create a model from a Modelfile
  show        Show model information
  run         Run a model
  stop        Stop a running model
  pull        Pull/import a model
  push        Push a model to a registry
  list        List models
  ls          Alias for list
  ps          List running models
  cp          Copy a model
  rm          Remove a model
  generate    Generate text
  embeddings  Generate embeddings
  signin      Sign in to a registry
  signout     Sign out
  launch      Launch an integration
  help        Show help
  version     Show version

Environment:
  CAPYBARA_MODEL
  CAPYBARA_CONTEXT
  CAPYBARA_BATCH
  CAPYBARA_THREADS
  CAPYBARA_GPU_LAYERS
  CAPYBARA_PARALLEL
  CAPYBARA_HOST
  CAPYBARA_PORT
  CAPYBARA_REGISTRY_URL

API:
  Ollama compatible:
    http://127.0.0.1:11434/api

  OpenAI compatible:
    http://127.0.0.1:11434/v1
"""
    )


def main():
    parser = argparse.ArgumentParser(
        add_help=False,
        prog="capybara"
    )

    parser.add_argument(
        "-v",
        "--version",
        action="store_true"
    )

    parser.add_argument(
        "command",
        nargs="?"
    )

    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER
    )

    parsed = parser.parse_args()

    if parsed.version:
        command_version(None)
        return

    command = parsed.command

    if not command:
        help_text()
        return

    aliases = {
        "ls": "list",
        "help": "help"
    }

    command = aliases.get(
        command,
        command
    )

    if command == "help":
        help_text()
        return

    if command == "version":
        command_version(None)
        return

    if command == "serve":
        parser2 = argparse.ArgumentParser(
            prog="capybara serve"
        )

        parser2.add_argument(
            "extra",
            nargs="*"
        )

        args = parser2.parse_args(parsed.args)

        command_serve(args)
        return

    if command == "run":
        parser2 = argparse.ArgumentParser(
            prog="capybara run"
        )

        parser2.add_argument(
            "model",
            nargs="?"
        )

        parser2.add_argument(
            "prompt",
            nargs="?"
        )

        args = parser2.parse_args(parsed.args)

        command_run(args)
        return

    if command == "stop":
        parser2 = argparse.ArgumentParser(
            prog="capybara stop"
        )

        parser2.add_argument(
            "model",
            nargs="?"
        )

        args = parser2.parse_args(parsed.args)

        command_stop(args)
        return

    if command in {
        "list",
        "ps"
    }:
        if command == "list":
            command_list(None)
        else:
            command_ps(None)

        return

    if command == "rm":
        parser2 = argparse.ArgumentParser(
            prog="capybara rm"
        )

        parser2.add_argument(
            "model"
        )

        command_rm(
            parser2.parse_args(parsed.args)
        )

        return

    if command == "show":
        parser2 = argparse.ArgumentParser(
            prog="capybara show"
        )

        parser2.add_argument(
            "model"
        )

        command_show(
            parser2.parse_args(parsed.args)
        )

        return

    if command == "cp":
        parser2 = argparse.ArgumentParser(
            prog="capybara cp"
        )

        parser2.add_argument("source")
        parser2.add_argument("destination")

        command_cp(
            parser2.parse_args(parsed.args)
        )

        return

    if command == "pull":
        parser2 = argparse.ArgumentParser(
            prog="capybara pull"
        )

        parser2.add_argument("model")

        command_pull(
            parser2.parse_args(parsed.args)
        )

        return

    if command == "push":
        parser2 = argparse.ArgumentParser(
            prog="capybara push"
        )

        parser2.add_argument("model")

        command_push(
            parser2.parse_args(parsed.args)
        )

        return

    if command == "create":
        parser2 = argparse.ArgumentParser(
            prog="capybara create"
        )

        parser2.add_argument(
            "name"
        )

        parser2.add_argument(
            "-f",
            "--file",
            default="Modelfile"
        )

        command_create(
            parser2.parse_args(parsed.args)
        )

        return

    if command == "generate":
        parser2 = argparse.ArgumentParser(
            prog="capybara generate"
        )

        parser2.add_argument(
            "model"
        )

        parser2.add_argument(
            "prompt"
        )

        command_generate(
            parser2.parse_args(parsed.args)
        )

        return

    if command == "embeddings":
        parser2 = argparse.ArgumentParser(
            prog="capybara embeddings"
        )

        parser2.add_argument(
            "model"
        )

        parser2.add_argument(
            "text"
        )

        command_embeddings(
            parser2.parse_args(parsed.args)
        )

        return

    if command == "signin":
        parser2 = argparse.ArgumentParser(
            prog="capybara signin"
        )

        parser2.add_argument(
            "--token"
        )

        command_signin(
            parser2.parse_args(parsed.args)
        )

        return

    if command == "signout":
        command_signout(None)
        return

    if command == "launch":
        parser2 = argparse.ArgumentParser(
            prog="capybara launch"
        )

        parser2.add_argument(
            "integration"
        )

        parser2.add_argument(
            "--model"
        )

        command_launch(
            parser2.parse_args(parsed.args)
        )

        return

    die(
        f"unknown command '{command}'. "
        "Run 'capybara help'."
    )


if __name__ == "__main__":
    main()
PY

    chmod +x "$CAPYBARA_BIN/capybara"
}

install_to_path() {
    mkdir -p "$INSTALL_BIN"

    ln -sf "$CAPYBARA_BIN/capybara" \
        "$INSTALL_BIN/capybara"

    ln -sf "$SERVER" \
        "$INSTALL_BIN/llama-server"

    export PATH="$INSTALL_BIN:$PATH"

    case "${SHELL##*/}" in
        zsh)
            RC="$HOME/.zshrc"
            ;;
        bash)
            RC="$HOME/.bashrc"
            ;;
        fish)
            mkdir -p "$HOME/.config/fish"

            if ! fish -c "fish_add_path '$INSTALL_BIN'" \
                >/dev/null 2>&1; then
                true
            fi

            return
            ;;
        *)
            RC="$HOME/.profile"
            ;;
    esac

    touch "$RC"

    if ! grep -Fq "$INSTALL_BIN" "$RC"; then
        printf '\nexport PATH="%s:$PATH"\n' \
            "$INSTALL_BIN" >> "$RC"
    fi
}

create_uninstaller() {
    cat > "$CAPYBARA_BIN/capybara-uninstall" <<'EOF'
#!/usr/bin/env bash

set -e

HOME_DIR="${CAPYBARA_HOME:-$HOME/.capybara}"
INSTALL_BIN="${CAPYBARA_INSTALL_BIN:-$HOME/.local/bin}"

rm -f "$INSTALL_BIN/capybara"
rm -f "$INSTALL_BIN/llama-server"

rm -rf "$HOME_DIR"

echo "Capybara removed."
echo "Your shell PATH entry was left unchanged."
EOF

    chmod +x "$CAPYBARA_BIN/capybara-uninstall"

    ln -sf \
        "$CAPYBARA_BIN/capybara-uninstall" \
        "$INSTALL_BIN/capybara-uninstall"
}

create_api_info() {
    cat > "$CAPYBARA_HOME/api.json" <<EOF
{
  "ollama": "http://127.0.0.1:11434/api",
  "openai": "http://127.0.0.1:11434/v1"
}
EOF
}

main() {
    detect_os

    mkdir -p \
        "$CAPYBARA_HOME" \
        "$CAPYBARA_BIN" \
        "$CAPYBARA_MODELS" \
        "$CAPYBARA_RUN"

    detect_gpu

    log "GPU backend: $BACKEND"
    log "GPU: $GPU_NAME"

    install_dependencies
    clone_engine
    build_engine
    create_config
    create_cli
    install_to_path
    create_uninstaller
    create_api_info

    echo
    echo "Capybara installed."
    echo
    echo "Backend: $BACKEND"
    echo "GPU: $GPU_NAME"
    echo "Binary: $INSTALL_BIN/capybara"
    echo "Models: $CAPYBARA_MODELS"
    echo
    echo "Run:"
    echo "  capybara help"
    echo
    echo "Example:"
    echo "  capybara pull https://example.com/model.gguf"
    echo "  capybara run model"
    echo
    echo "APIs:"
    echo "  http://127.0.0.1:11434/api"
    echo "  http://127.0.0.1:11434/v1"
}

main "$@"

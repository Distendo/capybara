#!/usr/bin/env bash
set -euo pipefail

CAPYBARA_HOME="${CAPYBARA_HOME:-$HOME/.capybara}"
INSTALL_BIN="${CAPYBARA_INSTALL_BIN:-$HOME/.local/bin}"
SRC="$CAPYBARA_HOME/llama.cpp"
BIN="$CAPYBARA_HOME/bin"
MODELS="$CAPYBARA_HOME/models"
RUN="$CAPYBARA_HOME/run"

say(){ printf '[capybara] %s\n' "$1"; }
die(){ printf '[capybara] error: %s\n' "$1" >&2; exit 1; }
has(){ command -v "$1" >/dev/null 2>&1; }
cores(){ if has nproc; then nproc; elif has sysctl; then sysctl -n hw.ncpu; else echo 4; fi; }

OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
  Darwin) OS_NAME=macOS;;
  Linux) OS_NAME=Linux;;
  FreeBSD) OS_NAME=FreeBSD;;
  *) die "unsupported OS: $OS";;
esac

mkdir -p "$CAPYBARA_HOME" "$BIN" "$MODELS" "$RUN" "$INSTALL_BIN"

install_deps(){
  say "Installing dependencies"
  case "$OS_NAME" in
    macOS)
      has brew || die "Homebrew is required on macOS: https://brew.sh"
      brew install git cmake ninja pkg-config python3 curl jq >/dev/null 2>&1 || true
      ;;
    Linux)
      if has apt-get; then
        sudo apt-get update
        sudo apt-get install -y git cmake ninja-build build-essential pkg-config python3 python3-tk curl jq
      elif has dnf; then
        sudo dnf install -y git cmake ninja-build gcc gcc-c++ make pkgconf-pkg-config python3 python3-tk curl jq
      elif has pacman; then
        sudo pacman -Sy --needed --noconfirm git cmake ninja base-devel pkgconf python tk curl jq
      elif has zypper; then
        sudo zypper --non-interactive install git cmake ninja gcc gcc-c++ make pkg-config python3 python3-tk curl jq
      else
        die "no supported Linux package manager found"
      fi
      ;;
    FreeBSD)
      sudo pkg update
      sudo pkg install -y git cmake ninja gcc pkgconf python3 py311-tkinter curl jq || true
      ;;
  esac
}

detect_backend(){
  BACKEND=cpu
  GPU="CPU"
  if [[ "$OS_NAME" == macOS && "$ARCH" == arm64 ]]; then BACKEND=metal; GPU="Apple Metal"; return; fi
  if has nvidia-smi; then BACKEND=cuda; GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"; GPU="${GPU:-NVIDIA CUDA}"; return; fi
  if has rocminfo || has hipconfig; then BACKEND=rocm; GPU="AMD ROCm/HIP"; return; fi
  if has sycl-ls; then BACKEND=sycl; GPU="Intel SYCL"; return; fi
  if has vulkaninfo; then BACKEND=vulkan; GPU="Vulkan"; return; fi
}

# ---------------------------------------------------------------------------
# Prebuilt engine: llama.cpp publishes ready-to-run binaries for common
# platforms. Downloading one takes ~30 seconds instead of a ~10 min compile.
# ---------------------------------------------------------------------------
latest_tag(){
  local json tag=""
  if has jq; then
    tag="$(curl -fsSL --max-time 20 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1' \
      | jq -r '.[0].tag_name // empty')" || true
  else
    json="$(curl -fsSL --max-time 20 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1')" || true
    tag="$(printf '%s' "$json" | grep -o '"tag_name": *"[^"]*"' | head -n1 | sed 's/.*"tag_name": *"//;s/"$//')"
  fi
  printf '%s' "$tag"
}

prebuilt_asset(){
  # $1 = release tag; echoes the asset filename or nothing when unavailable.
  local t="$1"
  case "${OS_NAME}:${ARCH}:${BACKEND}" in
    macOS:arm64:*)          echo "llama-${t}-bin-macos-arm64.tar.gz";;
    macOS:x86_64:*)         echo "llama-${t}-bin-macos-x64.tar.gz";;
    Linux:x86_64:vulkan)    echo "llama-${t}-bin-ubuntu-vulkan-x64.tar.gz";;
    Linux:x86_64:cpu)       echo "llama-${t}-bin-ubuntu-x64.tar.gz";;
    Linux:aarch64:cpu)      echo "llama-${t}-bin-ubuntu-arm64.tar.gz";;
    *)                      :;;  # cuda/rocm/sycl need a tailored source build
  esac
}

verify_engine(){
  # The engine must run standalone (dylib rpaths are self-contained).
  if "$BIN/llama-server" --version >/dev/null 2>&1; then
    return 0
  fi
  say "Prebuilt engine failed to start on this machine"
  rm -f "$BIN/llama-server" "$BIN"/libggml*.dylib "$BIN"/libggml*.so* \
        "$BIN"/libllama*.dylib "$BIN"/libllama*.so* "$BIN"/libmtmd*.dylib "$BIN"/libmtmd*.so*
  return 1
}

try_prebuilt_engine(){
  if ! has curl; then
    say "curl not found; cannot download prebuilt engine"
    return 1
  fi
  local tag asset url stage srcdir
  tag="$(latest_tag)"
  if [[ -z "$tag" ]]; then
    say "Could not resolve latest llama.cpp release"
    return 1
  fi
  asset="$(prebuilt_asset "$tag")"
  if [[ -z "$asset" ]]; then
    say "No prebuilt engine for $OS_NAME/$ARCH/$BACKEND - building from source"
    return 1
  fi
  url="https://github.com/ggml-org/llama.cpp/releases/download/${tag}/${asset}"
  stage="$(mktemp -d)"
  say "Downloading prebuilt llama.cpp ${asset}"
  if ! curl -fL --retry 4 --retry-delay 2 --progress-bar -o "$stage/engine.tar.gz" "$url"; then
    say "Download failed (${url})"
    rm -rf "$stage"
    return 1
  fi
  mkdir -p "$stage/unpacked"
  if ! tar xzf "$stage/engine.tar.gz" -C "$stage/unpacked"; then
    say "Extraction failed"
    rm -rf "$stage"
    return 1
  fi
  srcdir="$(find "$stage/unpacked" -maxdepth 1 -type d -name 'llama-*' | head -n1)"
  if [[ -z "$srcdir" || ! -x "$srcdir/llama-server" ]]; then
    say "Unexpected archive layout"
    rm -rf "$stage"
    return 1
  fi
  cp "$srcdir/llama-server" "$BIN/llama-server"
  chmod +x "$BIN/llama-server"
  if [[ -x "$srcdir/llama-cli" ]]; then
    cp "$srcdir/llama-cli" "$BIN/llama-cli"
    chmod +x "$BIN/llama-cli"
  fi
  # Shared libraries live next to the binaries (@loader_path / $ORIGIN rpath).
  find "$srcdir" -maxdepth 1 \( -name 'lib*.dylib' -o -name 'lib*.so*' \) \
    -exec cp {} "$BIN/" \;
  rm -rf "$stage"
  verify_engine
}

install_engine_source(){
  if [[ -d "$SRC/.git" ]]; then
    say "Updating llama.cpp"
    git -C "$SRC" fetch --depth=1 origin master >/dev/null 2>&1 || true
    git -C "$SRC" reset --hard origin/master >/dev/null 2>&1 || true
  else
    say "Downloading llama.cpp sources"
    git clone --depth=1 https://github.com/ggml-org/llama.cpp.git "$SRC"
  fi

  BUILD="$SRC/build-$BACKEND"
  rm -rf "$BUILD"
  args=( -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_BUILD_SERVER=ON )
  case "$BACKEND" in
    metal) args+=( -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON );;
    cuda) args+=( -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_ENABLE_UNIFIED_MEMORY=ON );;
    rocm) args+=( -DGGML_HIP=ON );;
    sycl) args+=( -DGGML_SYCL=ON );;
    vulkan) args+=( -DGGML_VULKAN=ON );;
    cpu) args+=( -DGGML_OPENMP=ON -DGGML_CPU_ALL_VARIANTS=ON );;
  esac
  say "Building $BACKEND backend from source (this can take ~10 minutes)"
  cmake "${args[@]}"
  cmake --build "$BUILD" --config Release --parallel "$(cores)"
  server="$BUILD/bin/llama-server"
  [[ -x "$server" ]] || die "llama-server build failed"
  cp "$server" "$BIN/llama-server"
  chmod +x "$BIN/llama-server"
  if [[ -x "$BUILD/bin/llama-cli" ]]; then
    cp "$BUILD/bin/llama-cli" "$BIN/llama-cli"
    chmod +x "$BIN/llama-cli"
  fi
}

install_engine(){
  if [[ "${CAPYBARA_ENGINE_SOURCE:-0}" == "1" ]]; then
    say "CAPYBARA_ENGINE_SOURCE=1 set - building from source"
    install_engine_source
    return
  fi
  if try_prebuilt_engine; then
    say "Engine installed from prebuilt binaries ($BACKEND)"
    return
  fi
  install_deps
  install_engine_source
}

install_python(){
  cp "$(dirname "$0")/capybara.py" "$BIN/capybara.py"
  cp "$(dirname "$0")/capybara.py" "$CAPYBARA_HOME/capybara.py"
  cp "$(dirname "$0")/server.py" "$CAPYBARA_HOME/server.py"
  mkdir -p "$CAPYBARA_HOME/ui"
  cp "$(dirname "$0")/ui/index.html" "$CAPYBARA_HOME/ui/index.html"
  if [[ -f "$(dirname "$0")/gui.py" ]]; then
    cp "$(dirname "$0")/gui.py" "$CAPYBARA_HOME/gui.py"
  fi
  chmod +x "$BIN/capybara.py"

  cat > "$INSTALL_BIN/capybara" <<EOF2
#!/usr/bin/env bash
exec "$BIN/capybara.py" "\$@"
EOF2
  chmod +x "$INSTALL_BIN/capybara"

  cat > "$INSTALL_BIN/capybara-gui" <<EOF2
#!/usr/bin/env bash
exec "$(command -v python3 || echo python3)" "$CAPYBARA_HOME/gui.py" "\$@"
EOF2
  chmod +x "$INSTALL_BIN/capybara-gui"

  ln -sf "$BIN/llama-server" "$INSTALL_BIN/llama-server"
}

write_env(){
  cat > "$CAPYBARA_HOME/config" <<EOF2
CAPYBARA_HOME=$CAPYBARA_HOME
CAPYBARA_MODELS=$MODELS
CAPYBARA_BACKEND=$BACKEND
CAPYBARA_GPU=$GPU
CAPYBARA_HOST=127.0.0.1
CAPYBARA_PORT=11434
EOF2

  export PATH="$INSTALL_BIN:$PATH"
  case "${SHELL##*/}" in
    zsh) rc="$HOME/.zshrc";;
    bash) rc="$HOME/.bashrc";;
    fish) fish -c "fish_add_path '$INSTALL_BIN'" >/dev/null 2>&1 || true; rc="";;
    *) rc="$HOME/.profile";;
  esac
  if [[ -n "${rc:-}" ]]; then
    touch "$rc"
    if ! grep -Fq "$INSTALL_BIN" "$rc" 2>/dev/null; then
      # $PATH is written literally (unexpanded) into the rc file on purpose.
      # shellcheck disable=SC2016
      printf '\nexport PATH="%s:$PATH"\n' "$INSTALL_BIN" >> "$rc"
    fi
  fi
}

main(){
  say "Detected $OS_NAME $ARCH"
  detect_backend
  say "Backend: $BACKEND ($GPU)"

  if [[ "${CAPYBARA_ENGINE_ONLY:-0}" != "1" ]] && ! has python3; then
    install_deps
  fi

  install_engine
  if [[ "${CAPYBARA_ENGINE_ONLY:-0}" == "1" ]]; then
    say "Engine-only install requested; skipping CLI setup"
    return
  fi
  install_python
  write_env
  echo
  echo "Capybara installed."
  echo "CLI:     capybara"
  echo "GUI:     capybara ui        (opens the built-in web app)"
  echo "Models:  $MODELS"
  echo "API:     http://127.0.0.1:11434/v1"
  echo
  echo "Try:"
  echo "  capybara pull smollm          # tiny 135M model, fast download"
  echo "  capybara serve                # web UI at http://localhost:11434"
  echo "  capybara run smollm 'hi!'     # one-shot prompt"
  echo "  capybara ui                   # chat in your browser"
  echo
  echo "Optional config: $CAPYBARA_HOME/config.yaml"
}
main "$@"

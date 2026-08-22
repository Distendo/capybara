#!/usr/bin/env bash
set -euo pipefail

CAPYBARA_HOME="${CAPYBARA_HOME:-$HOME/.capybara}"
INSTALL_BIN="${CAPYBARA_INSTALL_BIN:-$HOME/.local/bin}"
SRC="$CAPYBARA_HOME/llama.cpp"
BIN="$CAPYBARA_HOME/bin"
MODELS="$CAPYBARA_HOME/models"
RUN="$CAPYBARA_HOME/run"
GUI="$CAPYBARA_HOME/gui.py"

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
        sudo dnf install -y git cmake ninja-build gcc gcc-c++ make pkgconf-pkg-config python3 python3-tkinter curl jq
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

install_engine(){
  if [[ -d "$SRC/.git" ]]; then
    say "Updating llama.cpp"
    git -C "$SRC" fetch --depth=1 origin master >/dev/null 2>&1 || true
    git -C "$SRC" reset --hard origin/master >/dev/null 2>&1 || true
  else
    say "Downloading llama.cpp"
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
  say "Building $BACKEND backend"
  cmake "${args[@]}"
  cmake --build "$BUILD" --config Release --parallel "$(cores)"
  server="$BUILD/bin/llama-server"
  [[ -x "$server" ]] || die "llama-server build failed"
  cp "$server" "$BIN/llama-server"
  chmod +x "$BIN/llama-server"
  [[ -x "$BUILD/bin/llama-cli" ]] && cp "$BUILD/bin/llama-cli" "$BIN/llama-cli" && chmod +x "$BIN/llama-cli" || true
}

install_python(){
  cp "$(dirname "$0")/capybara.py" "$BIN/capybara.py"
  cp "$(dirname "$0")/gui.py" "$CAPYBARA_HOME/gui.py"
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
    grep -Fq "$INSTALL_BIN" "$rc" 2>/dev/null || printf '\nexport PATH="%s:$PATH"\n' "$INSTALL_BIN" >> "$rc"
  fi
}

main(){
  say "Detected $OS_NAME $ARCH"
  detect_backend
  say "Backend: $BACKEND ($GPU)"
  install_deps
  install_engine
  install_python
  write_env
  echo
  echo "Capybara installed."
  echo "CLI: capybara"
  echo "GUI: capybara-gui"
  echo "Models: $MODELS"
  echo "Ollama API: http://127.0.0.1:11434/api"
  echo "OpenAI API: http://127.0.0.1:11434/v1"
  echo
  echo "Try:"
  echo "  capybara pull tensorblock/SmolLM2-135M-Instruct-GGUF:Q2_K"
  echo "  capybara run SmolLM2-135M-Instruct-Q2_K.gguf"
  echo "  capybara-gui"
}
main "$@"

#!/usr/bin/env bash
# Capybara installer - Ollama-style local model runner
#
#   curl -fsSL https://raw.githubusercontent.com/Distendo/capybara/main/install.sh | bash
#
# Works three ways:
#   1. run from a repo checkout     -> installs local files
#   2. piped from curl              -> downloads files from GitHub
#   3. CAPYBARA_ENGINE_ONLY=1       -> only installs the llama.cpp engine
#
# Options: --help  --source  --engine-only  --cli-only  --uninstall [--purge]  -y
# Env: CAPYBARA_HOME  CAPYBARA_INSTALL_BIN  CAPYBARA_RAW  NO_COLOR

set -Eeuo pipefail

CAPYBARA_HOME="${CAPYBARA_HOME:-$HOME/.capybara}"
INSTALL_BIN="${CAPYBARA_INSTALL_BIN:-$HOME/.local/bin}"
RC_MANAGE=1
if [[ -n "${CAPYBARA_INSTALL_BIN:-}" ]]; then RC_MANAGE=0; fi
SRC="$CAPYBARA_HOME/llama.cpp"
BIN="$CAPYBARA_HOME/bin"
MODELS="$CAPYBARA_HOME/models"
RUN="$CAPYBARA_HOME/run"
LOG="$CAPYBARA_HOME/install.log"

REPO_RAW="${CAPYBARA_RAW:-https://raw.githubusercontent.com/Distendo/capybara/main}"

FLAG_ENGINE_ONLY="${CAPYBARA_ENGINE_ONLY:-0}"
FLAG_SOURCE=0 FLAG_CLI_ONLY=0 FLAG_PURGE=0 FLAG_YES=0
ACTION=install
STAGE=""

mkdir -p "$CAPYBARA_HOME" 2>/dev/null || true

# ---------------------------------------------------------------- TUI plumbing
if [[ -t 1 && "${TERM:-dumb}" != dumb && -z "${NO_COLOR:-}" ]]; then
  T_DIM=$'\e[2m'    T_BOLD=$'\e[1m'
  T_RED=$'\e[31m'   T_GREEN=$'\e[32m'  T_YEL=$'\e[33m'
  T_CYAN=$'\e[36m'  T_MAG=$'\e[35m'    T_0=$'\e[0m'
else
  T_DIM='' T_BOLD='' T_RED='' T_GREEN='' T_YEL='' T_CYAN='' T_MAG='' T_0=''
fi
case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in
  *UTF*8*|*utf*8*) UTF8=1 ;;
  *) UTF8=0 ;;
esac
G_OK='+' G_NO='x' G_STEP='*'
if (( UTF8 )); then
  G_OK="$(printf '\342\234\224')"   # check mark
  G_NO="$(printf '\342\234\226')"   # heavy x
  G_STEP="$(printf '\342\227\206')" # diamond
fi

SPIN_PID=""
FRAMES=("-" "\\" "|" "/")
if (( UTF8 )); then
  # braille spinner built from raw UTF-8 bytes (bash 3.2 has no \u escapes)
  FRAMES=("$(printf '\342\240\213')" "$(printf '\342\240\231')" \
          "$(printf '\342\240\271')" "$(printf '\342\240\270')" \
          "$(printf '\342\240\274')" "$(printf '\342\240\264')" \
          "$(printf '\342\241\246')" "$(printf '\342\241\247')")
fi

cleanup(){
  spin_stop
  if [[ -n "$STAGE" ]]; then rm -rf "$STAGE"; fi
}
trap cleanup EXIT
trap 'rc=$?; spin_stop; printf "\r%b[%s]%b unexpected failure at line %s (exit %s)\n" \
      "$T_RED" "$G_NO" "$T_0" "$LINENO" "$rc" >&2; exit "$rc"' ERR

has(){ command -v "$1" >/dev/null 2>&1; }

rep(){ printf '%*s' "$2" '' | tr ' ' "$1"; }

center_pad(){
  local s="$1" w="$2"
  local l=$(( (w - ${#s}) / 2 ))
  if (( l < 0 )); then l=0; fi
  printf '%s%s%s' "$(rep ' ' "$l")" "$s" "$(rep ' ' "$(( w - l - ${#s} ))")"
}

banner(){
  local t="C A P Y B A R A" s="local LLM runtime"
  local inner=$(( ${#s} + 4 ))
  if (( inner < ${#t} )); then inner=${#t}; fi
  local bar
  bar="   ${T_DIM}+$(rep - "$inner")+${T_0}"
  printf '\n'
  printf '%b\n' "$bar"
  printf '%b\n' "   ${T_DIM}|${T_0}${T_BOLD}$(center_pad "$t" "$inner")${T_0}${T_DIM}|${T_0}"
  printf '%b\n' "   ${T_DIM}|${T_0}${T_MAG}$(center_pad "$s" "$inner")${T_0}${T_DIM}|${T_0}"
  printf '%b\n' "$bar"
  printf '\n'
}

section(){ printf '\n%b %s %b%b%s%b\n' "$T_DIM" "$G_STEP" "$T_0" "$T_BOLD" "$1" "$T_0"; }

task(){ printf '   %-34s ' "$1"; }

ok(){ printf '%b%s%b %b%s%b\n' "$T_GREEN" "$G_OK" "$T_0" "$T_DIM" "${1:-done}" "$T_0"; }
bad(){ printf '%b%s%b %b%s%b\n' "$T_RED" "$G_NO" "$T_0" "$T_RED" "${1:-failed}" "$T_0"; }
note(){ printf '     %b%s%b\n' "$T_DIM" "$1" "$T_0"; }
warn(){ printf '   %b!%b %s\n' "$T_YEL" "$T_0" "$1"; }
die(){ bad "$1"; note "full log: $LOG"; exit 1; }

spin_start(){
  [[ -t 2 ]] || return 0
  (
    local i=0 t0=$SECONDS f
    while :; do
      f="${FRAMES[$((i++ % ${#FRAMES[@]}))]}"
      printf '\r     %b%s%b %bs%b' "$T_CYAN" "$f" "$T_0" "$T_DIM" "$((SECONDS - t0))$T_0" >&2
      sleep 0.08
    done
  ) &
  SPIN_PID=$!
}
spin_stop(){
  if [[ -n "$SPIN_PID" ]]; then
    kill "$SPIN_PID" 2>/dev/null || true
    wait "$SPIN_PID" 2>/dev/null || true
    SPIN_PID=''
    if [[ -t 2 ]]; then printf '\r     \r' >&2; fi
  fi
}

run_quiet(){
  local label="$1"; shift
  task "$label"
  spin_start
  if "$@" >>"$LOG" 2>&1; then
    spin_stop; ok
  else
    local rc=$?
    spin_stop; bad
    die "'$label' failed (exit $rc)"
  fi
}

fetch(){
  local url="$1" dest="$2" label="$3"
  task "$label"
  if [[ -t 2 ]]; then
    if curl -fL --retry 4 --retry-delay 2 --progress-bar -o "$dest" "$url"; then
      ok "$(du -h "$dest" | cut -f1)"
    else
      bad; die "download failed: $url"
    fi
  else
    spin_start
    if curl -fsSL --retry 4 --retry-delay 2 -o "$dest" "$url" >>"$LOG" 2>&1; then
      spin_stop; ok
    else
      spin_stop; bad; die "download failed: $url"
    fi
  fi
}

try_fetch_optional(){
  local url="$1" dest="$2" label="$3"
  task "$label"
  if curl -fsSL --retry 2 --max-time 30 -o "$dest" "$url" >>"$LOG" 2>&1; then
    ok
  else
    printf '%b-%b %b%s%b\n' "$T_DIM" "$T_0" "$T_DIM" "skipped" "$T_0"
    rm -f "$dest"
    return 1
  fi
}

panel(){
  local title="$1"; shift
  local rows=("$@") r key val pad
  local wide=${#title}
  for r in "${rows[@]}"; do
    key="${r%%|*}"; val="${r#*|}"
    if (( ${#key} + ${#val} > wide )); then wide=$(( ${#key} + ${#val} )); fi
  done
  # Row layout: " key dots val " = 3 fixed spaces + >=1 dot,
  # so the box must be 4 wider than the longest key|value pair.
  local inner=$(( wide + 4 ))
  printf '\n'
  printf '%b\n' "  ${T_DIM}+$(rep - "$inner")+${T_0}"
  printf '%b\n' "  ${T_DIM}|${T_0}${T_BOLD}$(center_pad "$title" "$inner")${T_0}${T_DIM}|${T_0}"
  printf '%b\n' "  ${T_DIM}+$(rep - "$inner")+${T_0}"
  for r in "${rows[@]}"; do
    key="${r%%|*}"; val="${r#*|}"
    pad=$(( inner - 3 - ${#key} - ${#val} ))
    printf '%b\n' "  ${T_DIM}|${T_0} ${T_BOLD}${key}${T_0}$(rep '.' "$pad") ${val} ${T_DIM}|${T_0}"
  done
  printf '%b\n' "  ${T_DIM}+$(rep - "$inner")+${T_0}"
  printf '\n'
}

usage(){
  cat <<EOF
Capybara installer

Usage: [env vars] install.sh [options]

Options:
  -h, --help         show this help
  --source           force building llama.cpp from source
  --engine-only      install only the llama.cpp engine
  --cli-only         install only the Python CLI/server (reuse existing engine)
  --ref REF          install sources from branch/tag REF instead of main
  --uninstall        remove capybara (keeps downloaded models)
  --purge            remove capybara INCLUDING all downloaded models
  -y, --yes          assume yes; never prompt

Env:
  CAPYBARA_HOME          install root (default ~/.capybara)
  CAPYBARA_INSTALL_BIN   wrapper dir added to PATH (default ~/.local/bin)
  CAPYBARA_RAW           raw-content base URL override
  CAPYBARA_ENGINE_ONLY=1 same as --engine-only
  NO_COLOR=1             disable colors
EOF
}

parse_args(){
  while (( $# )); do
    case "$1" in
      -h|--help)      usage; exit 0 ;;
      --source)       FLAG_SOURCE=1 ;;
      --engine-only)  FLAG_ENGINE_ONLY=1 ;;
      --cli-only)     FLAG_CLI_ONLY=1 ;;
      --uninstall)    ACTION=uninstall ;;
      --purge)        ACTION=uninstall; FLAG_PURGE=1 ;;
      --ref)          shift; REPO_RAW="https://raw.githubusercontent.com/Distendo/capybara/$1" ;;
      -y|--yes)       FLAG_YES=1 ;;
      *)              die "unknown option: $1 (see --help)" ;;
    esac
    shift
  done
}

confirm(){
  (( FLAG_YES )) && return 0
  [[ -t 0 ]] || return 0
  local reply
  printf '   Proceed? [%sy%s/N] ' "$T_BOLD" "$T_0"
  read -r reply
  [[ "${reply:-y}" == y || "${reply:-y}" == Y ]]
}

# --------------------------------------------------------------- locate sources
SOURCE_MODE=remote
SRC_DIR=''
resolve_sources(){
  local cand dir
  for cand in "${BASH_SOURCE[0]:-}" "$PWD/install.sh"; do
    [[ -n "$cand" && "$cand" != '-bash' && "$cand" != 'bash' && "$cand" != /dev/stdin* && "$cand" != '(stdin)' && "$cand" != stdin* ]] || continue
    dir="$(cd "$(dirname "$cand")" 2>/dev/null && pwd)" || continue
    if [[ -f "$dir/capybara.py" && -f "$dir/server.py" ]]; then
      SRC_DIR="$dir"; SOURCE_MODE=local; return 0
    fi
  done
  return 0
}

fetch_sources(){
  STAGE="$(mktemp -d)"
  local f
  for f in capybara.py server.py; do
    fetch "$REPO_RAW/$f" "$STAGE/$f" "fetch $f"
  done
  try_fetch_optional "$REPO_RAW/gui.py" "$STAGE/gui.py" "fetch gui.py" || true
  [[ -f "$STAGE/server.py" && -f "$STAGE/capybara.py" ]] || die "could not download capybara sources"
  SRC_DIR="$STAGE"
}

# ------------------------------------------------------------------- engine bits
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
  Darwin)  OS_NAME=macOS ;;
  Linux)   OS_NAME=Linux ;;
  FreeBSD) OS_NAME=FreeBSD ;;
  *) die "unsupported OS: $OS" ;;
esac

cores(){ if has nproc; then nproc; elif has sysctl; then sysctl -n hw.ncpu; else echo 4; fi; }

detect_backend(){
  BACKEND=cpu
  GPU="CPU only"
  if [[ "$OS_NAME" == macOS && "$ARCH" == arm64 ]]; then
    BACKEND=metal; GPU="Apple Silicon (Metal)"; return
  fi
  if has nvidia-smi; then
    BACKEND=cuda
    GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
    GPU="${GPU:-NVIDIA CUDA}"; return
  fi
  if has rocminfo || has hipconfig; then BACKEND=rocm;  GPU="AMD ROCm/HIP"; return; fi
  if has sycl-ls;    then BACKEND=sycl;  GPU="Intel SYCL";    return; fi
  if has vulkaninfo; then BACKEND=vulkan; GPU="Vulkan";        return; fi
}

latest_tag(){
  local json tag=""
  if has jq; then
    tag="$(curl -fsSL --max-time 20 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1' \
      | jq -r '.[0].tag_name // empty')" || true
  elif has python3; then
    tag="$(curl -fsSL --max-time 20 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1' \
      | python3 -c 'import json,sys;print(json.load(sys.stdin)[0]["tag_name"])' 2>/dev/null)" || true
  else
    json="$(curl -fsSL --max-time 20 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1')" || true
    tag="$(printf '%s' "$json" | grep -o '"tag_name": *"[^"]*"' | head -n1 | sed 's/.*"tag_name": *"//;s/"$//')"
  fi
  printf '%s' "$tag"
}

prebuilt_asset(){
  local t="$1"
  case "${OS_NAME}:${ARCH}:${BACKEND}" in
    macOS:arm64:*)       echo "llama-${t}-bin-macos-arm64.tar.gz" ;;
    macOS:x86_64:*)      echo "llama-${t}-bin-macos-x64.tar.gz" ;;
    Linux:x86_64:vulkan) echo "llama-${t}-bin-ubuntu-vulkan-x64.tar.gz" ;;
    Linux:x86_64:cpu)    echo "llama-${t}-bin-ubuntu-x64.tar.gz" ;;
    Linux:aarch64:cpu)   echo "llama-${t}-bin-ubuntu-arm64.tar.gz" ;;
    *)                   : ;;
  esac
}

verify_engine(){
  if "$BIN/llama-server" --version >/dev/null 2>&1; then
    return 0
  fi
  note "prebuilt engine failed to start here; falling back to source build"
  rm -f "$BIN/llama-server" "$BIN"/libggml*.dylib "$BIN"/libggml*.so* \
        "$BIN"/libllama*.dylib "$BIN"/libllama*.so* \
        "$BIN"/libmtmd*.dylib "$BIN"/libmtmd*.so*
  return 1
}

try_prebuilt_engine(){
  if ! has curl; then note "curl missing; cannot download prebuilt engine"; return 1; fi
  local tag asset url stage srcdir
  task "resolve llama.cpp release"
  spin_start
  tag="$(latest_tag)"
  spin_stop
  if [[ -z "$tag" ]]; then
    bad "no release found"; note "will build from source"; return 1
  fi
  ok "$tag"

  asset="$(prebuilt_asset "$tag")"
  if [[ -z "$asset" ]]; then
    note "no prebuilt binary for $OS_NAME/$ARCH/$BACKEND - building from source"
    return 1
  fi

  url="https://github.com/ggml-org/llama.cpp/releases/download/${tag}/${asset}"
  stage="$(mktemp -d)"
  mkdir -p "$stage/unpacked"
  fetch "$url" "$stage/engine.tar.gz" "download engine"
  run_quiet "unpack engine" tar xzf "$stage/engine.tar.gz" -C "$stage/unpacked"
  srcdir="$(find "$stage/unpacked" -maxdepth 1 -type d -name 'llama-*' | head -n1)"
  if [[ -z "$srcdir" || ! -x "$srcdir/llama-server" ]]; then
    note "unexpected archive layout - building from source"
    rm -rf "$stage"
    return 1
  fi

  mkdir -p "$BIN"
  cp "$srcdir/llama-server" "$BIN/llama-server"
  chmod +x "$BIN/llama-server"
  if [[ -x "$srcdir/llama-cli" ]]; then
    cp "$srcdir/llama-cli" "$BIN/llama-cli"
    chmod +x "$BIN/llama-cli"
  fi
  find "$srcdir" -maxdepth 1 \( -name 'lib*.dylib' -o -name 'lib*.so*' \) -exec cp {} "$BIN/" \;
  rm -rf "$stage"
  verify_engine
}

run_soft(){
  local label="$1"; shift
  task "$label"
  spin_start
  if "$@" >>"$LOG" 2>&1; then
    spin_stop; ok
  else
    spin_stop
    printf '%b-%b %bcontinued (see %s)%b\n' "$T_DIM" "$T_0" "$T_DIM" "$LOG" "$T_0"
  fi
}

install_deps(){
  case "$OS_NAME" in
    macOS)
      has brew || die "Homebrew is required on macOS: https://brew.sh"
      run_soft "brew install tools" brew install git cmake ninja pkg-config python3 curl jq
      ;;
    Linux)
      if has apt-get; then
        run_quiet "apt install tools" sudo apt-get update
        run_quiet "apt install tools" sudo apt-get install -y git cmake ninja-build build-essential pkg-config python3 python3-tk curl jq
      elif has dnf; then
        run_quiet "dnf install tools" sudo dnf install -y git cmake ninja-build gcc gcc-c++ make pkgconf-pkg-config python3 python3-tk curl jq
      elif has pacman; then
        run_quiet "pacman install tools" sudo pacman -Sy --needed --noconfirm git cmake ninja base-devel pkgconf python tk curl jq
      elif has zypper; then
        run_quiet "zypper install tools" sudo zypper --non-interactive install git cmake ninja gcc gcc-c++ make pkg-config python3 python3-tk curl jq
      else
        die "no supported Linux package manager found"
      fi
      ;;
    FreeBSD)
      run_soft "pkg update" sudo pkg update
      run_soft "pkg install tools" sudo pkg install -y git cmake ninja gcc pkgconf python3 py311-tkinter curl jq
      ;;
  esac
}

install_engine_source(){
  if [[ -d "$SRC/.git" ]]; then
    run_quiet "update llama.cpp" bash -c "git -C '$SRC' fetch --depth=1 origin master && git -C '$SRC' reset --hard origin/master"
  else
    run_quiet "clone llama.cpp" git clone --depth=1 https://github.com/ggml-org/llama.cpp.git "$SRC"
  fi

  local BUILD="$SRC/build-$BACKEND"
  rm -rf "$BUILD"
  local args=( -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_BUILD_SERVER=ON )
  case "$BACKEND" in
    metal) args+=( -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON ) ;;
    cuda)  args+=( -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_ENABLE_UNIFIED_MEMORY=ON ) ;;
    rocm)  args+=( -DGGML_HIP=ON ) ;;
    sycl)  args+=( -DGGML_SYCL=ON ) ;;
    vulkan) args+=( -DGGML_VULKAN=ON ) ;;
    cpu)   args+=( -DGGML_OPENMP=ON -DGGML_CPU_ALL_VARIANTS=ON ) ;;
  esac

  task "configure ($BACKEND)"
  spin_start
  if cmake "${args[@]}" >>"$LOG" 2>&1; then spin_stop; ok; else
    local rc=$?; spin_stop; bad; die "cmake configure failed"
  fi

  task "compile ($(cores) cores)"
  spin_start
  if cmake --build "$BUILD" --config Release --parallel "$(cores)" >>"$LOG" 2>&1; then
    spin_stop; ok
  else
    local rc=$?; spin_stop; bad; die "build failed after several minutes - see $LOG"
  fi

  local server="$BUILD/bin/llama-server"
  [[ -x "$server" ]] || die "llama-server build produced no binary"
  mkdir -p "$BIN"
  cp "$server" "$BIN/llama-server"
  chmod +x "$BIN/llama-server"
  if [[ -x "$BUILD/bin/llama-cli" ]]; then
    cp "$BUILD/bin/llama-cli" "$BIN/llama-cli"
    chmod +x "$BIN/llama-cli"
  fi
  run_quiet "smoke test engine" "$BIN/llama-server" --version
}

existing_engine_ok(){
  [[ -x "$BIN/llama-server" ]] && "$BIN/llama-server" --version >/dev/null 2>&1
}

install_engine(){
  if (( FLAG_SOURCE )); then
    note "--source given: building llama.cpp from source"
    install_deps
    install_engine_source
    return
  fi
  if (( FLAG_CLI_ONLY )); then
    if existing_engine_ok; then
      task "existing engine"; ok "kept"
      return
    fi
    warn "--cli-only requested but no working engine found; fetching one"
  fi
  if try_prebuilt_engine; then
    return
  fi
  install_deps
  install_engine_source
}

# ------------------------------------------------------------------ CLI install
check_python(){
  if ! has python3; then
    note "python3 missing - installing dependencies"
    install_deps
  fi
  task "python version"
  if python3 - <<'PY' 2>>"$LOG"
import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)
PY
  then
    ok "$(python3 -V)"
  else
    bad; die "Python 3.9+ is required (found $(python3 -V 2>&1))"
  fi
}

install_python(){
  mkdir -p "$BIN" "$RUN" "$MODELS" "$INSTALL_BIN"
  task "install CLI + server"
  cp "$SRC_DIR/capybara.py" "$BIN/capybara.py"
  cp "$SRC_DIR/capybara.py" "$CAPYBARA_HOME/capybara.py"
  cp "$SRC_DIR/server.py"   "$CAPYBARA_HOME/server.py"
  if [[ -f "$SRC_DIR/gui.py" ]]; then
    cp "$SRC_DIR/gui.py" "$CAPYBARA_HOME/gui.py"
  fi
  chmod +x "$BIN/capybara.py"
  ok

  task "create launcher"
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
  ok "$INSTALL_BIN/capybara"

  if python3 -c 'import tkinter' 2>>"$LOG"; then
    task "gui toolkit (tkinter)"; ok
  else
    task "gui toolkit (tkinter)"
    printf '%b-%b %b%s%b\n' "$T_DIM" "$T_0" "$T_DIM" "absent - browser UI still works" "$T_0"
  fi
}

write_env(){
  task "default config"
  cat > "$CAPYBARA_HOME/config" <<EOF2
CAPYBARA_HOME=$CAPYBARA_HOME
CAPYBARA_MODELS=$MODELS
CAPYBARA_BACKEND=$BACKEND
CAPYBARA_GPU=$GPU
CAPYBARA_HOST=127.0.0.1
CAPYBARA_PORT=11434
EOF2
  ok

  export PATH="$INSTALL_BIN:$PATH"
  local rc_file=""
  if (( ! RC_MANAGE )); then
    task "PATH entry"
    printf '%b-%b %b%s%b\n' "$T_DIM" "$T_0" "$T_DIM" "custom CAPYBARA_INSTALL_BIN - add it to PATH yourself" "$T_0"
    return
  fi
  case "${SHELL##*/}" in
    zsh)  rc_file="$HOME/.zshrc" ;;
    bash) rc_file="$HOME/.bashrc" ;;
    fish)
      run_quiet "register PATH (fish)" fish -c "fish_add_path '$INSTALL_BIN'"
      ;;
    *)    rc_file="$HOME/.profile" ;;
  esac
  if [[ -n "$rc_file" ]]; then
    task "PATH entry"
    touch "$rc_file"
    if ! grep -Fq "$INSTALL_BIN" "$rc_file" 2>/dev/null; then
      # written literally (unexpanded) on purpose
      # shellcheck disable=SC2016
      printf '\nexport PATH="%s:$PATH"\n' "$INSTALL_BIN" >> "$rc_file"
      ok "$rc_file"
    else
      printf '%b-%b %b%s%b\n' "$T_DIM" "$T_0" "$T_DIM" "already present" "$T_0"
    fi
  fi
}

finish_panel(){
  panel "installed" \
    "cli|$INSTALL_BIN/capybara" \
    "accel|$GPU" \
    "backend|$BACKEND" \
    "models|$MODELS" \
    "api|http://127.0.0.1:11434" \
    "log|$LOG"
  printf '%bNext:%b\n' "$T_BOLD" "$T_0"
  printf '   %bcapybara pull smollm%b      %b# tiny model, fast download%b\n' "$T_CYAN" "$T_0" "$T_DIM" "$T_0"
  printf '   %bcapybara run smollm "hi"%b  %b# chat in your terminal%b\n' "$T_CYAN" "$T_0" "$T_DIM" "$T_0"
  printf '   %bcapybara serve%b            %b# Ollama-compatible API on :11434%b\n' "$T_CYAN" "$T_0" "$T_DIM" "$T_0"
  printf '   %bcapybara ui%b               %b# chat in your browser%b\n' "$T_CYAN" "$T_0" "$T_DIM" "$T_0"
  printf '\n'
}

# -------------------------------------------------------------------- uninstall
do_uninstall(){
  banner
  section "Removing Capybara"
  if (( FLAG_PURGE )); then
    warn "this deletes ALL downloaded models in $MODELS"
    confirm || { note "aborted"; exit 0; }
  fi
  run_quiet "remove launchers" rm -f "$INSTALL_BIN/capybara" "$INSTALL_BIN/capybara-gui" "$INSTALL_BIN/llama-server"
  run_quiet "remove engine"    rm -rf "$BIN"
  run_quiet "remove runtime"   rm -rf "$RUN" "$CAPYBARA_HOME/config"
  rm -f "$CAPYBARA_HOME/capybara.py" "$CAPYBARA_HOME/server.py" "$CAPYBARA_HOME/gui.py"
  if (( FLAG_PURGE )); then
    run_quiet "remove models"  rm -rf "$MODELS"
    run_quiet "remove sources" rm -rf "$SRC" "$CAPYBARA_HOME/config.yaml"
  else
    task "keep models"; ok "$MODELS"
  fi
  if (( FLAG_PURGE )); then
    panel "removed" \
      "note|all files under $CAPYBARA_HOME were deleted" \
      "note|PATH entries in your shell rc were left alone"
  else
    panel "removed" \
      "kept models|$MODELS" \
      "note|PATH entries in your shell rc were left alone"
  fi
}

# ------------------------------------------------------------------------- main
main(){
  parse_args "$@"
  : > "$LOG"

  if [[ "$ACTION" == uninstall ]]; then
    do_uninstall
    return
  fi

  banner
  resolve_sources

  section "System"
  task "platform"; ok "$OS_NAME $ARCH"
  detect_backend
  task "accelerator"; ok "$GPU"
  task "cores"; ok "$(cores)"

  if [[ "$SOURCE_MODE" == local ]]; then
    task "sources"; ok "from checkout ($SRC_DIR)"
  else
    task "sources"; ok "remote ($REPO_RAW)"
  fi

  if [[ "$SOURCE_MODE" == remote && "$FLAG_ENGINE_ONLY" != 1 ]]; then
    section "Fetching Capybara"
    fetch_sources
  fi

  section "Engine"
  install_engine

  if [[ "$FLAG_ENGINE_ONLY" == 1 ]]; then
    section "Done"
    task "scope"; ok "engine only (--engine-only)"
    panel "engine installed" \
      "engine|$BIN/llama-server" \
      "accel|$GPU"
    return
  fi

  section "CLI"
  check_python
  install_python
  write_env

  finish_panel
}

main "$@"

# Capybara

[![CI](https://github.com/Distendo/capybara/actions/workflows/ci.yml/badge.svg)](https://github.com/Distendo/capybara/actions/workflows/ci.yml)

![Capybara screenshot](img/2026-08-21_18.35.43.png.webp)

# Ollama, but much better.

## About

Capybara is a local model runner inspired by Ollama, and much better than Ollama.

It is built to keep the basic workflow simple:

```
capybara pull llama3
capybara run llama3
```

Models run locally on your machine and can also be exposed through an
OpenAI-compatible HTTP API.

Under the hood Capybara drives [llama.cpp](https://github.com/ggml-org/llama.cpp)
as its inference engine — you get state-of-the-art GGUF inference with Metal,
CUDA, ROCm, SYCL or Vulkan acceleration without having to touch a single build
flag yourself.

## Install

Build it from source (macOS, Linux and FreeBSD; requires Python 3.9+):

```
git clone https://github.com/Distendo/capybara.git
cd capybara
make build        # or: ./install.sh
```

The installer detects your hardware, builds the right llama.cpp backend,
and installs `capybara` / `capybara-gui` to `~/.local/bin`.

Only need the engine? `make engine` skips CLI setup.

## Usage

| Command | Description |
| --- | --- |
| `capybara pull <model>` | download a model (alias, HF repo, URL or local file) |
| `capybara run <model> [prompt]` | chat interactively, or one-shot with a prompt |
| `capybara list` | list installed models |
| `capybara inspect <model>` | show details about a model (`show` works too) |
| `capybara rm <model>` | remove a model |
| `capybara cp <src> <dst>` | copy a model under a new name |
| `capybara serve [--model M] [-F]` | start the API server (`-F` = foreground) |
| `capybara ps` | server status |
| `capybara stop` | stop the API server |
| `capybara logs [-n N]` | tail the engine log |
| `capybara create -f Modelfile name` | create a model from a Modelfile |
| `capybara launch <prog> --model M` | run a program wired to the local API |

Running a model that isn't installed pulls it automatically first.

### Chat commands

Inside `capybara run`:

```
/bye     exit (also /exit, /quit, Ctrl-D)
/clear   reset the conversation
/help    list commands
```

## Models

Pull by alias, Hugging Face repo, exact quantization, direct URL or local path:

```
capybara pull smollm                          # tiny test model (~80 MB)
capybara pull llama3                          # alias -> curated repo + Q4_K_M
capybara pull bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M
capybara pull https://example.com/model.gguf  # direct URL
```

Sharded (multi-part) GGUF releases are detected and downloaded completely.
Quantization defaults to Q4_K_M when a repo offers it.

Built-in aliases:

| Alias | Model |
| --- | --- |
| `smollm` | SmolLM2 135M Instruct (Q2_K) |
| `llama3` | Meta Llama 3 8B Instruct |
| `llama3.1` | Meta Llama 3.1 8B Instruct |
| `qwen2.5` | Qwen 2.5 7B Instruct |
| `mistral` | Mistral 7B Instruct v0.3 |
| `gemma2` | Gemma 2 9B IT |
| `phi3` | Phi-3 mini 4k Instruct |
| `deepseek-r1` | DeepSeek R1 Distill Qwen 7B |

List what you have:

```
$ capybara list
NAME                      SIZE      MODIFIED
SmolLM2-135M-Instruct.Q2_K  84.2 MB  2026-08-22 12:40
```

## API

Capybara serves an OpenAI-compatible API on:

```
http://localhost:11434/v1
```

Example:

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [
      {
        "role": "user",
        "content": "Hello"
      }
    ]
  }'
```

It can therefore be used with applications that already support
OpenAI-compatible endpoints.

### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="capybara"
)

response = client.chat.completions.create(
    model="llama3",
    messages=[
        {"role": "user", "content": "Hello"}
    ]
)

print(response.choices[0].message.content)
```

## Configuration

Optional config file at `~/.capybara/config.yaml`:

```yaml
runtime:
  threads: 8          # CPU threads (default: all cores)
  context: 8192       # context window (default: 10240)
  gpu_layers: 999     # layers offloaded to GPU (default: all)
  batch: 2048
  ubatch: 512

server:
  host: 127.0.0.1
  port: 11434

models:
  directory: ~/.capybara/models
```

Precedence: defaults < `config.yaml` < environment variables.
Every key has a matching env override: `CAPYBARA_HOST`, `CAPYBARA_PORT`,
`CAPYBARA_MODELS`, `CAPYBARA_THREADS`, `CAPYBARA_CONTEXT`,
`CAPYBARA_GPU_LAYERS`, `CAPYBARA_BATCH`, `CAPYBARA_UBATCH`,
`CAPYBARA_HOME`.

The configuration file is optional. Capybara works with sensible defaults
when it is not present.

### Modelfiles

`capybara create` supports Ollama-style Modelfiles:

```
FROM llama3
PARAMETER temperature 0.8
PARAMETER num_ctx 8192
SYSTEM You are a helpful capybara living in the terminal.
```

```
capybara create -f Modelfile my-capy
capybara run my-capy
```

System prompt and sampling parameters are applied at inference time;
`num_ctx` is applied when the engine starts for that model.

## Hardware

Capybara is intended to run on normal consumer hardware.
The installer auto-detects the best available backend:

```
Apple Metal   (default on Apple Silicon)
CUDA          (NVIDIA GPUs, via nvidia-smi)
ROCm/HIP      (AMD GPUs)
SYCL          (Intel GPUs)
Vulkan
CPU           (fallback, always available)
```

## Architecture

Capybara separates the command line interface, API, model management and
inference backend.

```
                 Capybara
                    │
          ┌─────────┴─────────┐
          │                   │
         CLI                GUI/API
          │                   │
          └─────────┬─────────┘
                    │
              Model Manager
                    │
               Runtime (llama-server)
                    │
             Backend Layer
          ┌─────────┼─────────┐
          │         │         │
        Metal     CUDA      CPU
```

This makes it possible to add different inference backends without changing
the public interface.

## Project structure

```
capybara/
├── capybara.py        # single-file CLI: model manager + runtime controller
├── capybara.test.py   # unit tests (stdlib unittest, no dependencies)
├── gui.py             # tkinter desktop companion
├── install.sh         # backend detection + llama.cpp build + install
├── Makefile           # make build / test / dev / engine
└── .github/
    └── workflows/ci.yml
```

No Python packages required — everything runs on the standard library.

## Building

Clone the repository:

```
git clone https://github.com/Distendo/capybara.git
cd capybara
```

Build & install:

```
make build
```

Run tests:

```
make test
```

Run the development version without installing:

```
make dev ARGS="list"
```

---

## Status

Capybara is a working prototype: pulling, running, serving and Modelfiles all
work today. Expect rough edges; interfaces may still change before 1.0.

Windows is not supported yet.

## License

Apache-2.0

See [`LICENSE`](LICENSE) for the full license.

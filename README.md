# Capybara

Single-file local LLM runner. No backends, no agents, no GUI — just `pip install` and go.

```bash
pip install -e .
capybara pull bartowski/Meta-Llama-3-8B-Instruct-GGUF
capybara run llama3 "hi"
```

## Install

Python 3.9+

```bash
git clone https://github.com/Distendo/capybara.git
cd capybara
pip install -e .
```

Requires a `llama-server` binary on PATH or in `./bin/`.

## Usage

| Command | Description |
| --- | --- |
| `capybara pull <model>` | download from HF (or `ollama/name:tag`) |
| `capybara run <model> [prompt]` | chat interactively or one-shot |
| `capybara serve [--model M] [-F]` | start API + web UI (`-F` = foreground) |
| `capybara ui` | open Web UI in browser |
| `capybara list` / `ls` | list installed models |
| `capybara show <model>` | model details |
| `capybara rm <model>` | remove a model |
| `capybara cp <src> <dst>` | copy model |
| `capybara create -f Modelfile <name>` | create from Modelfile |
| `capybara search <query>` | search HuggingFace for GGUF models |
| `capybara ps` | server status |
| `capybara stop` | stop server |
| `capybara logs [-n N]` | tail engine log |

Running an uninstalled model pulls it automatically.

## Engine arguments

Pass native `llama-server` flags after `--`:

```bash
capybara run smollm -- --n-predict 512 --temp 0.7 --threads 8
```

## API

- OpenAI-compatible `POST /v1/chat/completions`
- Ollama-compatible `POST /api/chat`, `POST /api/generate`, `GET /api/tags`
- Built-in chat UI on the same port

## Build / Publish

```bash
python -m build
twine upload dist/*
```

No Python dependencies beyond the standard library.

## License

Apache-2.0
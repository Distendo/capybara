# Capybara

<p align="center">
  <img src="https://github.com/FoarteBine/capybara/img/blob/main/2026-08-21_18.35.43.png.webp?raw=true" width="160">
</p>

<p align="center">
  A local AI runtime for running and serving language models.
</p>

---

## About

Capybara is a local model runner inspired by Ollama.

It is built to keep the basic workflow simple:

```bash
capybara pull llama3
capybara run llama3
```

Models run locally on your machine and can also be exposed through an HTTP API.

---

## Install

Build it from source:

```bash
git clone https://github.com/FoarteBine/capybara.git
cd capybara

make build
```

---

## Usage

Download a model:

```bash
capybara pull llama3
```

Run it:

```bash
capybara run llama3
```

List installed models:

```bash
capybara list
```

Remove a model:

```bash
capybara rm llama3
```

Show model information:

```bash
capybara inspect llama3
```

Start the API server:

```bash
capybara serve
```

---

## API

Capybara provides an OpenAI-compatible API.

By default:

```text
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

It can therefore be used with applications that already support OpenAI-compatible endpoints.

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
        {
            "role": "user",
            "content": "Hello"
        }
    ]
)

print(response.choices[0].message.content)
```

---

## Models

Models are stored locally.

```bash
capybara list
```

Example:

```text
NAME             SIZE
llama3:8b        4.7 GB
qwen3:8b         5.1 GB
deepseek-r1      4.9 GB
```

Model storage can be changed through the configuration file.

---

## Configuration

Example configuration:

```yaml
runtime:
  threads: 8
  context: 8192
  gpu: true

server:
  host: 127.0.0.1
  port: 11434

models:
  directory: ~/.capybara/models
```

The configuration file is optional. Capybara works with sensible defaults when it is not present.

---

## Hardware

Capybara is intended to run on normal consumer hardware.

Supported acceleration depends on the backend and platform.

Currently planned backends include:

```text
CPU
CUDA
Metal
Vulkan
ROCm
```

CPU inference is available as a fallback.

---

## Architecture

Capybara separates the command line interface, API, model management and inference backend.

```text
                 Capybara
                    │
          ┌─────────┴─────────┐
          │                   │
         CLI                 API
          │                   │
          └─────────┬─────────┘
                    │
              Model Manager
                    │
               Runtime
                    │
             Backend Layer
          ┌─────────┼─────────┐
          │         │         │
         CPU       CUDA      Metal
```

This makes it possible to add different inference backends without changing the public interface.

---

## Project structure

```text
capybara/
(for now nothing)
```

---

## Building

Clone the repository:

```bash
git clone https://github.com/FoarteBine/capybara.git
cd capybara
```

Build:

```bash
make build
```

Run tests:

```bash
make test
```

Run the development version:

```bash
make dev
```

---

## Status

Capybara is currently under development.

The CLI, API and runtime interfaces may change before the first stable release.

---

## License

Apache-2.0

See [`LICENSE`](LICENSE) for the full license.

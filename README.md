# mimoe-edge-agent

A minimal conversational CLI agent that runs entirely on-device using the local
inference endpoint exposed by [mimOE Studio](https://mimik.com/mimoestudio/).
No cloud API keys required — the model runs locally, the inference endpoint is
local, and no data leaves the machine.

---

## What it does

- Connects to the mimOE local inference endpoint via the OpenAI-compatible API.
- Streams responses token by token so output appears immediately as the model generates it.
- Renders responses as live markdown in the terminal (bold, code blocks, lists) via `rich`.
- Maintains conversation history within a session (multi-turn dialogue) using a sliding
  window (last 10 turns) to prevent context window overflow on small models.
- Supports two interaction modes that can be switched at any time:
  - **chat** — general-purpose assistant
  - **code** — programming-focused assistant with a code-oriented system prompt
- Handles connection errors, API errors, and missing configuration gracefully, rolling
  back the user message from history if a request fails.
- Structured logging with configurable level via `LOG_LEVEL` environment variable.

---

## Prerequisites

- Python 3.10+
- [mimOE Studio](https://mimik.com/mimoestudio/) installed, running, and a model loaded (e.g. SmolLM2)

---

## Setup

**1. Clone the repo**

```bash
git clone <your-repo-url>
cd mimoe-edge-agent
```

**2. Create a virtual environment and install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The `[dev]` extras install `pytest`, `pytest-cov`, `ruff`, and `pre-commit`
alongside the runtime dependencies.

**3. Install pre-commit hooks**

```bash
pre-commit install
pre-commit install --hook-type pre-push
```

Ruff (lint + format) runs on every `git commit`. Tests run on every `git push` —
unit tests always, integration tests only when Studio is running.

**4. Configure the endpoint**

```bash
cp .env.example .env
```

Open `.env` and fill in the values from mimOE Studio:

- Open mimOE Studio → **Model View** → click the **API** button.
- The panel will show the base URL and model identifier.
- Set `MIMOE_BASE_URL` and `MIMOE_MODEL` to match what the panel shows.

```env
MIMOE_BASE_URL=http://localhost:8080/v1
MIMOE_MODEL=SmolLM2
MIMOE_API_KEY=not-required
```

**5. Run**

```bash
python main.py
```

---

## Usage

```
You: explain what an edge node is in two sentences

Agent (CHAT): An edge node is a device at the periphery of a network that
processes data locally rather than sending it to a central cloud server.
This reduces latency, improves privacy, and allows computation to continue
even without a reliable internet connection.

You: /code
Switched to CODE mode — Code-focused assistant — writes and explains code

You: write a Python function that retries a failing HTTP call with backoff

Agent (CODE): ...
```

### Available commands

| Command  | Effect                                            |
|----------|---------------------------------------------------|
| `/chat`  | Switch to general assistant mode                  |
| `/code`  | Switch to code assistant mode                     |
| `/clear` | Clear conversation history (current mode is kept) |
| `/mode`  | Show the current mode                             |
| `/help`  | Show command list                                 |
| `/quit`  | Exit                                              |

### Debug logging

Set `LOG_LEVEL=DEBUG` to see internal state traces (history rollbacks, system prompt
updates) without HTTP library noise:

```bash
LOG_LEVEL=DEBUG python main.py
```

---

## Development

**Run the unit tests**

```bash
pytest
```

**Run with coverage**

```bash
pytest --cov --cov-report=term-missing
```

**Run the integration tests**

Integration tests hit the live mimOE endpoint and require Studio to be running
with a model loaded:

```bash
pytest -m integration -v
```

If Studio is not running, all tests skip gracefully and the command exits 0.

**Run hooks manually**

```bash
pre-commit run --all-files                        # ruff only (commit-stage hooks)
pre-commit run --all-files --hook-stage pre-push  # tests (unit + integration)
```

---

## Project structure

```
mimoe-edge-agent/
├── agent/
│   ├── utils/
│   │   └── terminal.py      # TerminalFormatter + strip_latex for markdown rendering
│   ├── client.py            # builds the OpenAI client from .env config
│   ├── conversation.py      # in-session message history with sliding window
│   └── modes.py             # named system-prompt personas (CHAT / CODE)
├── tests/
│   ├── integration/
│   │   ├── conftest.py      # session fixtures + auto-skip when Studio is offline
│   │   └── test_endpoint.py # live endpoint tests (run with: pytest -m integration)
│   ├── test_client.py       # build_client / get_model startup and config tests
│   ├── test_conversation.py # Conversation history and sliding window tests
│   ├── test_main.py         # REPL loop, complete(), and setup_logging tests
│   ├── test_modes.py        # mode definitions and system prompt tests
│   └── test_terminal.py     # TerminalFormatter and strip_latex tests
├── main.py                  # CLI entry point and REPL loop
├── pyproject.toml           # package definition, dependencies, tool config
├── .pre-commit-config.yaml  # ruff + unit and integration pytest hooks
├── .env.example             # configuration template
├── .gitignore
└── README.md
```

---

## Design choices

### Thin OpenAI-compatible client, no agent framework

I used the OpenAI SDK as a lightweight client wrapper because mimOE exposes an OpenAI-compatible endpoint. I intentionally kept the integration thin instead of adding LangChain, LlamaIndex, or another agent framework.

For this scope, I wanted the mimOE connection to stay easy to follow: configure the local endpoint, send messages to the loaded model, stream the response back, and keep the agent logic transparent.

### Streaming responses

The chat completion call uses `stream=True` so tokens are written to the terminal as soon as the local model generates them.

This matters more with smaller local models because responses can feel slow if the terminal stays frozen until the full completion is done. Streaming makes the CLI feel more responsive while still allowing the app to assemble the full assistant reply and store it in conversation history.

### Rich markdown rendering

Responses are rendered as live markdown using `rich.Live` and `rich.Markdown`, which updates the display as tokens arrive. This means bold text, code blocks, and lists render properly rather than printing raw markup.

Small local models often emit LaTeX math notation regardless of instructions. A `strip_latex()` pre-processing step converts `$$...$$` display math to fenced code blocks and `$...$` inline math to backtick spans, which `rich` can render without errors.

### Sliding window context management

Conversation history is capped at the last 10 user/assistant pairs before each request. The system prompt is always included regardless of history length.

This prevents context window exhaustion on small models like SmolLM2, which typically support 2 048–4 096 tokens. Long sessions would otherwise degrade response quality or cause errors as the prompt grows. `max_tokens=512` on each request caps output length for the same reason.

### Mode switching as the agentic behavior

I kept the "agentic" behavior focused and explicit. Instead of trying to force tool calling or complex orchestration onto a small local model, the app supports mode switching through different system prompts.

The user can choose the active mode, and the agent carries that context through the conversation. This keeps the behavior predictable while still showing a real assistant pattern: routing the same model through different task contexts.

### Structured logging with a clean terminal format

I separated conversational output from operational output.

`print()` is only used for the actual chat experience: the `You:` prompt and the assistant's streamed replies. That output goes to stdout, which keeps it easy to read or pipe elsewhere.

Operational messages, such as startup status, mode changes, warnings, and errors, go through `logging` to stderr. INFO logs are formatted like clean status messages without a level prefix, while WARNING and ERROR messages include the level so problems stand out.

The `LOG_LEVEL` environment variable controls verbosity without requiring source changes.

### In-memory history only

Conversation history is stored in a Python list for the current session and discarded when the app exits.

I intentionally did not add persistence for this version. Saving history to SQLite, JSON, or another store would require decisions around schema, session resumption, truncation, and privacy. Those are valid next steps, but they were outside the focused scope of this assessment.

### Environment-based configuration

`MIMOE_BASE_URL`, `MIMOE_MODEL`, and `MIMOE_API_KEY` are loaded from `.env` at startup.

The app exits early with a clear error message if a required value is missing. This keeps the project portable because the exact endpoint and model name should come from the API panel in mimOE Studio rather than being hardcoded.

---

## Limitations

* **Model capability:** SmolLM2 and similarly small local models have more limited reasoning ability and context windows than larger hosted models. As conversation history grows, response quality may degrade even with the sliding window in place.
* **No tool use or function calling:** I intentionally left out tool calling because support can vary across small local models and OpenAI-compatible local runtimes.
* **No persistent history:** Conversation state only exists for the current CLI session.
* **API key placeholder:** mimOE's local endpoint does not appear to validate the API key, but the OpenAI SDK still requires a non-empty value. The app uses `"not-required"` as a safe local placeholder when no key is provided.

---

## Technical notes

### What I explored

I loaded SmolLM2 in mimOE Studio and used the built-in API panel to inspect the local OpenAI-compatible endpoint.

The API panel provided the key details needed by the client:

* the local `base_url`
* the model identifier
* example `curl` commands
* confirmation that the endpoint supports the standard chat completions route

After that, the integration was straightforward. The OpenAI Python client worked with the local mimOE endpoint by overriding `base_url`, so I did not need to write custom HTTP request logic.

Streaming also worked through the same client. With `stream=True`, the local server returned incremental response chunks, which the OpenAI SDK exposed as `ChatCompletionChunk` objects.

### Why this approach

I chose the OpenAI Python client because it is the simplest way to target an OpenAI-compatible endpoint without adding unnecessary abstraction.

The client already handles request construction, response parsing, streaming, retry behavior, and typed response objects. Pointing it at mimOE only requires changing the base URL and model configuration.

That felt like the right interpretation of the "BYO Framework" option: use a tool I already know, configure it for the local mimOE backend, and keep the rest of the app small enough to explain clearly.

### What "on-device AI" means in this project

In this project, the inference path stays local.

mimOE Studio acts as the local model server. It loads the model, manages the inference runtime, and exposes an HTTP API on the local machine. The CLI agent is just the client layer on top of that local server.

So the flow is:

```text
CLI agent
→ local mimOE OpenAI-compatible endpoint
→ loaded local model
→ streamed response back to the terminal
```

There is no external model API call from the agent, no cloud inference request, and no API billing tied to each completion. The main tradeoff is that quality, speed, and context length depend on the local model and the machine running it.

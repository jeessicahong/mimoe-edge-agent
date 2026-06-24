# mimoe-edge-agent

A minimal conversational CLI agent that runs entirely on-device using the local
inference endpoint exposed by [mimOE Studio](https://mimik.com/mimoestudio/).
No cloud API keys required — the model runs locally, the inference endpoint is
local, and no data leaves the machine.

---

## What it does

- Connects to the mimOE local inference endpoint via the OpenAI-compatible API.
- Streams responses token by token so output appears immediately as the model generates it.
- Maintains conversation history within a session (multi-turn dialogue).
- Supports two interaction modes that can be switched at any time:
  - **chat** — general-purpose assistant
  - **code** — programming-focused assistant with a code-oriented system prompt
- Handles connection errors, API errors, and missing configuration gracefully.
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

The `[dev]` extras install `pytest`, `ruff`, and `pre-commit` alongside the runtime dependencies.

**3. Install pre-commit hooks**

```bash
pre-commit install
```

This wires up ruff (lint + format) and pytest to run automatically on every `git commit`.

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

Agent (chat): An edge node is a device at the periphery of a network that
processes data locally rather than sending it to a central cloud server.
This reduces latency, improves privacy, and allows computation to continue
even without a reliable internet connection.

You: /code
Switched to code mode -- Code-focused assistant -- writes and explains code

You: write a Python function that retries a failing HTTP call with backoff

Agent (code): ...
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

**Run the test suite**

```bash
pytest
```

27 tests covering `Conversation`, `Mode`, and config validation in `client.py`.

**Run pre-commit manually against all files**

```bash
pre-commit run --all-files
```

This runs ruff (lint + format) and pytest. The same hooks fire automatically on
every `git commit`.

---

## Project structure

```
mimoe-edge-agent/
├── agent/
│   ├── client.py        # builds the OpenAI client from .env config
│   ├── conversation.py  # in-session message history
│   └── modes.py         # named system-prompt personas (chat / code)
├── tests/
│   ├── test_client.py
│   ├── test_conversation.py
│   └── test_modes.py
├── main.py                   # CLI entry point and REPL loop
├── pyproject.toml            # package definition, dependencies, tool config
├── .pre-commit-config.yaml   # ruff + pytest hooks
├── .env.example              # configuration template
├── .gitignore
└── README.md
```

---

## Design choices

**Thin OpenAI-compatible client, no agent framework.**
I used the OpenAI SDK only as a transport/client wrapper because mimOE exposes an OpenAI-compatible endpoint. I intentionally avoided higher-level agent frameworks so the mimOE integration stays transparent.

**Streaming responses.**
`stream=True` on the completions call sends each token to the terminal as soon
as the model generates it. This removes the frozen-terminal problem with slower
local models and makes the agent feel responsive. The full reply is still
assembled and stored in conversation history normally.

**Mode switching as the agentic behavior.**
Rather than simulating tool calls or orchestration the model cannot reliably
execute with a small local LLM, the "agentic" dimension here is explicit
context routing: the user controls which system prompt is active, and the
agent carries that context through the full conversation. This is a real
pattern used in production assistants.

**Structured logging with a clean terminal format.**
`print()` is reserved for conversational output (the agent's replies and the
`You:` prompt) so it goes to stdout and can be piped independently. All
operational messages — startup info, mode switches, errors — go through
`logging` to stderr. INFO messages have no level prefix (they are status lines,
not log entries); WARNING and above show the level so problems stand out. The
`LOG_LEVEL` environment variable controls verbosity without any source edits.

**In-memory history only.**
Session history is kept in a Python list and discarded on exit. Persistence
(SQLite, a JSON file) was a deliberate non-requirement for this scope — adding
it would require choosing a schema and a resumption strategy, neither of which
the assessment asks for.

**Environment-based configuration.**
`MIMOE_BASE_URL`, `MIMOE_MODEL`, and `MIMOE_API_KEY` are loaded from `.env`
at startup. The app exits immediately with a clear, actionable error message if
either required variable is missing.

---

## Limitations

- **Model capability:** SmolLM2 and similarly small local models have limited
  context windows and reasoning ability. Long conversations may degrade in
  quality as history grows.
- **No tool use / function calling:** Local small models vary widely in their
  support for the `tools` parameter. Excluded intentionally.
- **No persistent history:** Conversation is lost on exit.
- **API key placeholder:** mimOE's local endpoint does not validate the API
  key, but the OpenAI client requires a non-empty string. `"not-required"` is
  used as a safe placeholder.

---

## Technical notes (assessment)

### What I explored

I loaded SmolLM2 in mimOE Studio and used the built-in API panel to inspect the
local OpenAI-compatible endpoint. The panel exposes:

- A `base_url` (e.g. `http://localhost:<port>/v1`)
- A model identifier string
- Example `curl` commands confirming the standard `/chat/completions` route

The endpoint accepted standard `openai` Python client calls with `base_url`
overridden — no custom HTTP code was needed. Streaming (`stream=True`) also
worked out of the box, with the server sending chunked responses that the
OpenAI client iterates as `ChatCompletionChunk` objects.

### Why this approach

The OpenAI Python client's `base_url` override is the canonical way to target
any OpenAI-compatible third-party endpoint. It handles request construction,
response parsing, retry logic, and type-safe response objects. Pointing it at
a local mimOE endpoint requires exactly two lines of configuration change and
zero custom networking code — which is exactly what "BYO Framework" means in
practice: bring the tool you know, configure it for the new backend.

### What "on-device AI" means here

The entire inference stack runs on the local machine. mimOE Studio acts as a
local model server — it loads the model weights, manages the inference runtime,
and exposes an HTTP API. The agent code is a thin client. There is no external
network call, no API billing, and inference requests are sent to the local mimOE endpoint.

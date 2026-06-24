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
cp .env.example .env   # Windows: copy .env.example .env
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
LOG_LEVEL=DEBUG python main.py   # Windows: $env:LOG_LEVEL="DEBUG"; python main.py
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

I used the OpenAI SDK as a lightweight client because mimOE exposes an OpenAI-compatible endpoint. I intentionally avoided adding LangChain, LlamaIndex, or another higher-level agent framework because it would have made the integration harder to reason about for very little benefit in this scope.

The goal here was to keep the mimOE path transparent: configure the local endpoint, send messages to the loaded model, stream the response back, and keep the agent logic easy to explain.

### Streaming responses

The chat completion call uses `stream=True`, so tokens are written to the terminal as soon as the local model generates them.

This makes a noticeable difference with smaller local models. Without streaming, the terminal can look frozen while the model is still generating. With streaming, the CLI feels much more responsive. The full assistant reply is still assembled from the streamed chunks and saved into conversation history after the response completes.

### Rich markdown rendering

Responses are rendered as live markdown using `rich.Live` and `rich.Markdown`, so formatting like bold text, lists, and code blocks displays cleanly instead of showing up as raw markdown.

One issue I ran into is that small local models may still emit LaTeX-style math even when asked not to. Since `rich.Markdown` does not handle all LaTeX cleanly, I added a small `strip_latex()` preprocessing step. It converts `$$...$$` display math into fenced code blocks and `$...$` inline math into backtick spans, which keeps the terminal output readable and avoids rendering errors.

### Context management

The SmolLM2 model I tested with has a 2048-token context window. That limit applies to the full request, including the system prompt, recent conversation history, the latest user message, and the generated response.

To keep requests bounded, the agent keeps the system prompt plus the most recent user/assistant turns instead of sending the entire conversation every time. The response is also capped with `max_tokens=512`, which reserves part of the context window for generation rather than filling it entirely with prompt history.

This is a simple sliding-window approach. It keeps the implementation predictable and avoids adding tokenizer or summarization dependencies for a focused assessment project.

The tradeoff is that turn count is only an approximation for token usage. Ten short turns and ten turns with long code blocks can consume very different amounts of context. If I were extending this further, I would move to a token-budgeted window instead, likely starting with a lightweight character-based estimate such as `len(content) // 4` before adding a tokenizer dependency.



### Explicit task routing

I kept the interaction model intentionally small. Rather than adding tool calling or orchestration for complexity’s sake, the CLI supports explicit task routing through slash commands.

Each mode maps to a different system prompt, so the same local model can be used in different contexts while keeping the behavior easy to inspect and explain.

This is not a fully autonomous agent but rather a focused local assistant that demonstrates the core integration points: local inference, configuration, streaming, bounded history, mode selection, and error handling. More complex orchestration would be a natural next step, but it was outside the scope of this assessment.


### Structured logging with a clean terminal format

I separated conversational output from operational output.

`print()` is reserved for the actual chat experience: the `You:` prompt and the assistant’s streamed replies. That output goes to stdout, so it stays clean and can be piped independently if needed.

Startup messages, mode switches, warnings, and errors go through `logging` to stderr. INFO logs are formatted like simple status lines without a level prefix, while WARNING and ERROR messages include the level so issues stand out.

`LOG_LEVEL` controls verbosity without requiring code changes.

The logging formatter and `strip_latex()` helper live in `agent/utils/terminal.py` instead of `main.py`. I moved them there so the REPL loop could stay focused on control flow, and so the terminal/display helpers could be tested independently from the full agent stack.

### Error recovery with history rollback

If a request fails because the endpoint is unavailable or the API returns an error, the user message that was already added to history is removed before the loop continues.

This keeps the conversation state consistent. Otherwise, the next user message would be treated as a follow-up to a turn that never actually completed, leaving the model with a user message but no assistant reply. That can make multi-turn behavior worse, especially with smaller local models.

The rollback is handled by `pop_last_user()` on the `Conversation` class. It only removes the trailing message if that message is a user message, so it is safe to call from the error handlers.

### Two-layer input validation

Empty input is blocked in two places.

The REPL handles the common case immediately:

```python
if not user_input:
    continue
```

That prevents empty or whitespace-only prompts from being added to history or sent to the model.

The `Conversation.add_user()` and `Conversation.add_assistant()` methods enforce the same rule at the data layer by raising `ValueError` for empty content. That second check is intentional. The REPL guard protects the normal path, while the `Conversation` validation keeps the message list valid if the class is used in tests, another entry point, or future refactors.

### Test strategy

Unit tests mock the external pieces: the OpenAI client, model identifier, stdin, and `rich.Live`. That keeps the tests fast and avoids requiring mimOE Studio or a live local endpoint.

Integration tests live under `tests/integration/` and use `pytest.mark.integration`. They are excluded from the default `pytest` run. A session-scoped fixture checks whether the live mimOE endpoint is reachable before the integration tests run, and skips them if the endpoint is unavailable.

`pytest-cov` is configured in `pyproject.toml` to measure coverage across the `agent/` package and `main.py`. The current unit suite reaches 100% line coverage on production code, while integration tests are intentionally kept separate from that coverage target.

### In-memory history only

Conversation history is stored in a Python list for the current session and discarded when the app exits.

I intentionally did not add persistence for this version. Saving history to SQLite, JSON, or another store would require decisions around schema, session resumption, truncation, and privacy. Those are valid next steps, but they were outside the focused scope of this assessment.

### Environment-based configuration

`MIMOE_BASE_URL`, `MIMOE_MODEL`, and `MIMOE_API_KEY` are loaded from `.env` at startup.

The app exits early with a clear error message if a required value is missing. This keeps the project portable because the exact endpoint and model name should come from the API panel in mimOE Studio rather than being hardcoded.

### Use of AI tools

I used Claude Code and ChatGPT during this assessment as AI-assisted development tools.

I treated both tools as pair-programming aids rather than sources of final truth. Generated code and documentation were reviewed, tested, and adjusted before being accepted.

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

### What I would improve next

If I were extending this beyond the assessment scope, I would focus on:

* token-budgeted context management instead of turn-count truncation
* optional persistent sessions, likely through a small local SQLite store
* clearer model health checks at startup
* stronger handling for empty or low-quality model responses
* optional support for additional local models exposed through mimOE
* a small tool-calling layer if the selected model/runtime supports it reliably

I intentionally left these out of the first version to keep the project focused on the core integration: local mimOE inference, streaming responses, conversation state, mode selection, and clean terminal behavior.

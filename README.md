# tinytasktree

[![License](https://img.shields.io/github/license/orion-arm-ai/tinytasktree)](LICENSE.txt)
[![CI](https://github.com/orion-arm-ai/tinytasktree/actions/workflows/ci.yml/badge.svg)](https://github.com/orion-arm-ai/tinytasktree/actions/workflows/ci.yml)

A tiny async task-tree orchestrator library for Python, behavior-tree inspired and LLM-ready.

## Why tinytasktree?

- Modular, composable task graph building blocks
- Behavior-tree inspired control flow with explicit success/failure semantics
- Async-first execution with local trace visualization

## Hello World

```python
from tinytasktree import Tree

tree = (
    Tree("HelloWorld")
    .Sequence()
    ._().Function(lambda: "hello")
    .End()
)
```

## LLM Example

```python
from dataclasses import dataclass
from tinytasktree import Tree, JSON, Context

@dataclass
class Blackboard:
    prompt: str
    response: str = ""


def make_messages(b: Blackboard) -> list[JSON]:
    return [{"role": "user", "content": b.prompt}]


tree = (
    Tree[Blackboard]("HelloWorld")
    .Sequence()
    ._().LLM("openrouter/openai/gpt-4.1-mini", make_messages)
    ._().WriteBlackboard("response")
    .End()
)

async def main():
    context = Context()
    blackboard = Blackboard(prompt="Say hello in JSON.")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print(result)
    print(blackboard.response)
```

## Requirements

- Python 3.13+
- LiteLLM (only needed for `LLM` nodes)
- Redis (only needed for `Terminable` and `RedisCacher` nodes)
- Uvicorn (only needed for the HTTP trace server)

## Features

- Minimal, expressive tree builder API
- Async-first execution model
- Leaf / Composite / Decorator nodes built-in
- LLM integration via LiteLLM
- Redis-backed caching and termination signaling
- Trace collection and optional trace storage
- UI trace viewer with HTTP server

## API Stability <span id="api-stability"></span>

Alpha. Expect breaking changes until the API is stabilized.

## Installation

```bash
uv add tinytasktree
```

or

```bash
pip install tinytasktree
```

## UI Trace Server

Run the backend server and the React UI to view traces:

```bash
# 1) start backend
python -m tinytasktree --httpserver --host 127.0.0.1 --port 8000 --trace-dir .traces

# 2) start UI dev server
cd ui && npm run dev

# 3) open the UI
# http://127.0.0.1:5173
```

![](misc/tasktree-ui.png)

## Design Notes

- Execution model: nodes are awaited; composite nodes control child ordering and concurrency
- Results: nodes return `OK(data)` or `FAIL(data)` and composites propagate or short-circuit
- Blackboard: a per-run data object passed through the tree via `Context`

## Table of Contents <span id="ref"></span>

- [Node Reference](#node-reference)
  - [Node Result & Status](#node-result-status)
  - [Leaf Nodes](#leaf-nodes)
    - [Function](#function)
    - [Log](#log)
    - [TODO](#todo)
    - [ShowBlackboard](#showblackboard)
    - [WriteBlackboard](#writeblackboard)
    - [Assert](#assert)
    - [Failure](#failure)
    - [Subtree](#subtree)
    - [ParseJSON](#parsejson)
    - [LLM](#llm)
  - [Composite Nodes](#composite-nodes)
    - [Sequence](#sequence)
    - [Selector](#selector)
    - [Parallel](#parallel)
    - [Gather](#gather)
    - [RandomSelector](#randomselector)
    - [If / Else](#if--else)
  - [Decorator Nodes](#decorator-nodes)
    - [ForceOk](#forceok)
    - [ForceFail](#forcefail)
    - [Return](#return)
    - [Invert](#invert)
    - [Retry](#retry)
    - [While](#while)
    - [Timeout / Fallback](#timeout--fallback)
    - [RedisCacher](#rediscacher)
    - [Terminable](#terminable)
    - [Wrapper](#wrapper)
- [Core APIs (Non-Node)](#core-apis-non-node)
- [Contributing](#contributing)
- [License](#license)

## Node Reference <span id="node-reference"></span> <a href="#ref">[↑]</a>

### Node Result & Status <span id="node-result-status"></span>  <a href="#ref">[↑]</a>

- Every node returns a `Result` with a status (`OK` or `FAIL`) and an optional data payload
- Composite nodes typically short-circuit on `FAIL` (e.g., `Sequence`) or on `OK` (e.g., `Selector`)
- Decorators can override or invert status while preserving or transforming data

### Leaf Nodes <span id="leaf-nodes"></span> <a href="#ref">[↑]</a>

#### Function <span id="function"></span> <a href="#ref">[↑]</a>

Runs a sync/async function. Returns `OK(data)` for non-`Result` return values, or passes through a `Result`.

Usage:
- Accepts 0/1/2 params: (), (blackboard), (blackboard, tracer)
- Sync or async functions are supported
- Returning `Result` bypasses wrapping; otherwise `OK(value)`

Supported function forms:
- `() -> Any` or `() -> Result`
- `(blackboard) -> Any` or `(blackboard) -> Result`
- `(blackboard, tracer) -> Any` or `(blackboard, tracer) -> Result`
- Async variants of all the above

```python
tree = (
    Tree()
    .Sequence()
    ._().Function(lambda: "ok1")
    ._().Function(lambda blackboard: "ok2")
    ._().Function(lambda blackboard, tracer: "ok3")
    .End()
)
```

#### Log <span id="log"></span> <a href="#ref">[↑]</a>

Logs a message into the trace. Always returns `OK(None)`.

Usage:
- `msg_or_factory`: string or `(blackboard) -> str`
- `level`: trace level (default: info)
- Emits a trace log entry and continues

```python
tree = (
    Tree()
    .Sequence()
    ._().Log("hello step1")
    ._().Log(lambda b: f"hello, step2: job={b.job_id}", level="debug")
    .End()
)
```

#### TODO <span id="todo"></span> <a href="#ref">[↑]</a>

A placeholder node that always returns `OK(None)`.

Usage:
- No-op leaf for scaffolding or TODO spots
- Replace with real nodes later

```python
tree = (
    Tree()
    .Sequence()
    ._().TODO("Prepare the Params")
    ._().TODO("Call the LLM")
    ._().Function(real_step)
    ._().TODO("Collect the result")
    .End()
)
```

#### ShowBlackboard <span id="showblackboard"></span> <a href="#ref">[↑]</a>

Returns the current blackboard in `OK(b)`.

Usage:
- Helpful for debugging or inspection
- Downstream nodes can consume the returned blackboard

```python
tree = (
    Tree()
    .ShowBlackboard()
    .End()
)
```

#### WriteBlackboard <span id="writeblackboard"></span> <a href="#ref">[↑]</a>

Writes the previous node's result into the blackboard, and returns `OK(data)`.

Usage:
- `attr_or_func`: attribute name or `(blackboard, data) -> None`
- Reads `last_result.data`; warns if no last result
- Returns `OK(data)` (or `OK(None)` if missing)

```python
tree = (
    Tree()
    .Sequence()
    ._().Function(lambda: 123)
    ._().WriteBlackboard("value")
    .End()
)
```

or:

```python
def _set_value(b: Blackboard, v: int) -> None:
    b.double_value = v * 2

tree = (
    Tree()
    .Sequence()
    ._().Function(lambda: 7)
    ._().WriteBlackboard(_set_value)
    .End()
)
```

#### Assert <span id="assert"></span> <a href="#ref">[↑]</a>

Checks a boolean condition and returns `OK(True)` or `FAIL(False)`.

Usage:
- Condition can be attr name or function
- Sync/async; params 0/1/2: (), (blackboard), (blackboard, tracer)
- `AssertionError` is treated as false

```python
tree = (
    Tree()
    .Sequence()
    ._().Assert(lambda: True)
    ._().Assert("is_ready") # checks `blackboard.is_ready`
    ._().Function(run_job)
    .End()
)
```

#### Failure <span id="failure"></span> <a href="#ref">[↑]</a>

Always returns `FAIL(None)`.

Usage:
- Useful for tests, guards, or forcing failures

```python
tree = (
    Tree()
    .Selector()
    ._().Assert("has_cache")
    ._().Failure()
    .End()
)
```

#### Subtree <span id="subtree"></span> <a href="#ref">[↑]</a>

Runs another tree, optionally with a custom blackboard factory.

Usage:
- `subtree_blackboard_factory`: `(parent_blackboard) -> child_blackboard`
- Result is the subtree's result

```python
sub = (
    Tree()
    .Sequence()
    ._().Function(lambda: "x")
    .End()
)

tree = (
    Tree()
    .Sequence()
    ._().Subtree(sub) # or _().Subtree(sub, lambda b: SubBlackboard(b.text))
    .End()
)
```

#### ParseJSON <span id="parsejson"></span> <a href="#ref">[↑]</a>

Parses JSON from the last result or from a blackboard source, and returns the parsed object.

Usage:
- `src`: last result (default), blackboard attr, or `(blackboard) -> str`
- `dst`: optional blackboard attr or `(blackboard, data) -> None`
- Uses `orjson` with `json-repair` fallback and supports JSON code fences

```python
tree = (
    Tree()
    .Sequence()
    ._().Function(lambda: '{"a":1}')
    ._().ParseJSON(dst="data")
    .End()
)
```

another example:

```python
def set_parsed_value(b: blackboard, d: JSON) -> None:
    b.parsed = d

tree = (
    Tree()
    .Sequence()
    ._().ParseJSON(src="raw_json", dst=set_parsed_value)
    .End()
)
```

#### LLM <span id="llm"></span> <a href="#ref">[↑]</a>

Calls an LLM via LiteLLM and returns the output text. Supports streaming and API key factories.

Usage:
- `model` / `messages` can be values or `(blackboard) -> ...` factories
- `stream`: bool or `(blackboard) -> bool`; `stream_on_delta` supports sync/async callbacks
- `api_key`: string or factory `(blackboard)` / `(blackboard, model)`
- Tracer records tokens/cost when available

```python
tree = (
    Tree()
    .Sequence()
    ._().LLM("openrouter/openai/gpt-4.1-mini", [{"role": "user", "content": "hi"}])
    .End()
)
```

Streaming response example:

```python
def on_delta(b, full, delta, done, reason=""):
    if delta:
        print(delta, end="")

tree = (
    Tree()
    .Sequence()
    ._().LLM(lambda b: b.model, lambda b: b.messages, stream=True, stream_on_delta=on_delta)
    .End()
)
```

### Composite Nodes <span id="composite-nodes"></span> <a href="#ref">[↑]</a>

#### Sequence <span id="sequence"></span> <a href="#ref">[↑]</a>

Runs children in order. Returns `FAIL` on first failure, otherwise `OK(last_child_data)`.

Usage:
- Stops on first `FAIL` and returns last successful data
- Empty sequence returns `OK(None)`

```python
tree = (
    Tree()
    .Sequence()
    ._().Function(A)
    ._().Function(B)
    .End()
)
```

#### Selector <span id="selector"></span> <a href="#ref">[↑]</a>

Runs children in order until one succeeds. Returns the first `OK`, else `FAIL`.

Usage:
- Short-circuits on first success
- Empty selector returns `OK(None)`

```python
tree = (
    Tree()
    .Selector()
    ._().Failure()
    ._().Function(lambda: "ok")
    .End()
)
```

Selector is the typical choice for the fallback chain pattern:

```python
tree = (
    Tree()
    .Selector()
    ._().Timeout(20)
    ._()._().LLM("model1", llm_message)
    ._().Timeout(20)
    ._()._().LLM("model2", llm_message)
    ._().Timeout(20)
    ._()._().LLM("model3", llm_message)
    .End()
)
```

#### Parallel <span id="parallel"></span> <a href="#ref">[↑]</a>

Runs children concurrently. Returns `OK` only if all children succeed.

Usage:
- `concurrency_limit` must be > 0
- Result data is `None`

```python
tree = (
    Tree()
    .Parallel(concurrency_limit=2)
    ._().Function(A)
    ._().Function(B)
    .End()
)
```

#### Gather <span id="gather"></span> <a href="#ref">[↑]</a>

Runs multiple subtrees with their own blackboards and returns a list of results.

Usage:
- `params_factory`: `(blackboard) -> (trees, blackboards)`
- Runs each tree with its paired blackboard
- Returns list of child data in tree order

```python
tree = (
    Tree()
    .Gather(lambda b: (trees, blackboards))
    .End()
)
```

In a more detailed example:

```python
def build_params(b):
    trees = [subtree1, subtree2]
    bbs = [BB(x=1), BB(x=2)]
    return trees, bbs

tree = (
    Tree()
    .Gather(build_params, concurrency_limit=2)
    .End()
)
```

#### RandomSelector <span id="randomselector"></span> <a href="#ref">[↑]</a>

Randomizes the child order (optionally weighted) and returns the first `OK`.

Usage:
- `weights`: list or `(blackboard) -> list[float]`
- Weights must be positive and match child count

```python
tree = (
    Tree()
    .RandomSelector(weights=[0.4, 0.4, 0.2]) # or weights from a factory: lambda b: b.route_weights
    ._().Function(A)
    ._().Function(B)
    ._().Function(C)
    .End()
)
```

#### If / Else <span id="if--else"></span> <a href="#ref">[↑]</a>

Conditional branch. If the condition is false and no else branch exists, returns `OK(None)`.

Usage:
- Condition supports attr name or function (sync/async)
- 1 child (if) or 2 children (if + else)
- `Else` node must be a child of `If`

```python
tree = (
    Tree()
    .If(lambda b: b.flag)
    ._().Function(A)
    ._().Else()
    ._()._().Function(B)
    .End()
)
```

Or uses only `If` (without `Else`):

```python
tree = (
    Tree()
    .If("is_admin")
    ._().Function(admin_flow)
    .End()
)
```

### Decorator Nodes <span id="decorator-nodes"></span> <a href="#ref">[↑]</a>

#### ForceOk <span id="forceok"></span> <a href="#ref">[↑]</a>

Forces the result status to `OK`, optionally with a custom data factory.

Usage:
- Optional `result_factory(blackboard) -> data`
- If omitted, preserves child data

```python
tree = (
    Tree()
    .ForceOk()
    ._().Failure()
    .End()
)
```

Or a `ForceOk` overriding the result:

```python
tree = (
    Tree()
    .ForceOk(lambda b: {"skipped": True})
    ._().Function(best_effort)
    .End()
)
```

#### ForceFail <span id="forcefail"></span> <a href="#ref">[↑]</a>

Forces the result status to `FAIL`, optionally with a custom data factory.

Usage:
- Optional `result_factory(blackboard) -> data`
- If omitted, preserves child data

```python
tree = (
    Tree()
    .ForceFail()
    ._().Function(lambda: "x")
    .End()
)
```

#### Return <span id="return"></span> <a href="#ref">[↑]</a>

Preserves child status but replaces data with a factory result.

Usage:
- `result_factory(blackboard) -> data`
- Status is unchanged from child

```python
tree = (
    Tree()
    .Return(lambda b: "data")
    ._().Function(A)
    .End()
)
```

#### Invert <span id="invert"></span> <a href="#ref">[↑]</a>

Inverts child status while keeping data.

Usage:
- `OK` becomes `FAIL`, `FAIL` becomes `OK`
- Data is preserved

```python
tree = (
    Tree()
    .Invert()
    ._().Failure()
    .End()
)
```

#### Retry <span id="retry"></span> <a href="#ref">[↑]</a>

Retries a child on failure for up to `max_tries` with optional sleeps.

Usage:
- `sleep_secs`: float or list per retry index
- Returns first `OK`, else `FAIL(None)`

```python
tree = (
    Tree()
    .Retry(max_tries=3, sleep_secs=0.1) # or usage: [0.1, 0.2, 0.5]
    ._().Function(A)
    .End()
)
```

#### While <span id="while"></span> <a href="#ref">[↑]</a>

Repeats child while condition is true, returns the last successful result.

Usage:
- Condition supports attr name or function (sync/async)
- `max_loop_times` guards infinite loops

```python
tree = (
    Tree()
    .While(lambda b: b.count < 3)
    ._().Function(step)
    .End()
)
```

#### Timeout / Fallback <span id="timeout--fallback"></span> <a href="#ref">[↑]</a>

Runs a child with a time limit. On timeout, runs the fallback child if provided.

Usage:
- `Timeout` has 1 child (main) or 2 (main + fallback)
- On timeout, returns `FAIL(None)` or executes fallback
- `Fallback` node must be a child of `Timeout` or `Terminable`

```python
tree = (
    Tree()
    .Timeout(1.0)
    ._().Function(slow)
    ._().Function(on_timeout)
    .End()
)
```

With `.Fallback()` example::

```python
tree = (
    Tree()
    .Timeout(2.0)
    ._().Function(main_job)
    ._().Fallback()
    ._()._().Function(fallback_job)
    .End()
)
```

#### RedisCacher <span id="rediscacher"></span> <a href="#ref">[↑]</a>

Caches child results in Redis. Optional `value_validator` invalidates stale cache.

Usage:
- `key_func(blackboard) -> str`, optional `redis_client`
- `expiration`: seconds, `timedelta`, or random `(min, max)`
- `value_validator`: `(blackboard)` or `(blackboard, tracer)`
- `enabled`: bool or `(blackboard) -> bool`

```python
tree = (
    Tree()
    .RedisCacher(redis_client, key_func=lambda b: b.key, enabled=lambda b: b.use_cache)
    ._().Function(expensive_call)
    .End()
)
```

With a `value_validator` example, in such case: the cache is only considered a hit if this
value matches the one stored during the cache set. Useful for invalidating cache when dependent logic or state changes::

```python
tree = (
    Tree()
    .RedisCacher(redis_client, key_func=lambda b: f"user:{b.user_id}", value_validator=lambda b: b.version)
    ._().Function(fetch_user)
    .End()
)
```

#### Terminable <span id="terminable"></span> <a href="#ref">[↑]</a>

Runs a child while polling a Redis key for termination. Optionally runs a fallback.

Usage:
- `key_func(blackboard) -> redis_key`, optional `redis_client`
- Monitors until key exists; then cancels child
- 1 child (main) or 2 (main + fallback)

```python
tree = (
    Tree()
    .Terminable(lambda b: f"stop:{b.job_id}")
    ._().Function(A)
    ._().Fallback()
    ._()._().Function(B)
    .End()
)

# To trigger termination from an external script or process:
await redis.set(f"stop:{job_id}", "1")
```

#### Wrapper <span id="wrapper"></span> <a href="#ref">[↑]</a>

Wraps a child with a custom async context manager.

Usage:
- `func(child, context) -> async context manager` yielding a `Result`
- Useful for custom setup/teardown or instrumentation

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def traced(child, context):
    try:
        print("before run")
        result = await child(context)
        yield result
    finally:
        print("after run")

tree = (
    Tree()
    .Wrapper(traced)
    ._().Function(run)
    .End()
)
```

## Core APIs (Non-Node) <span id="core-apis-non-node"></span>

- `Context`: runtime state (blackboard stack, trace root, path)
- `TraceRoot` / `TraceNode`: structured trace tree
- `TraceStorageHandler` / `FileTraceStorageHandler`: save and load traces
- `register_global_hook_after_spawned_task_finish(hook)`: hook for Parallel/Gather/Terminable tasks
- `set_default_llm_api_key_factory(factory_or_key)`: default LLM API key or factory
- `set_default_global_redis_client(url, **kwargs)`: global Redis client for Redis nodes
- `run_httpserver(host, port, trace_dir)` / `create_http_app(...)`: HTTP trace server

## Contributing <span id="contributing"></span>

- Install dev dependencies: `uv sync --dev`
- Lint: `uv run ruff check .`
- Test: `uv run pytest` (requires a local Redis on `redis://127.0.0.1:6379`)

## License <span id="license"></span>

MIT. See `LICENSE.txt`.

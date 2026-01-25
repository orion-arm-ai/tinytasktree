# tinytasktree

A tiny async task-tree / behavior-tree style orchestrator for Python.

## Example

```python
from dataclasses import dataclass
from tinytasktree import Tree, JSON, Context

@dataclass
class Blackboard:
    prompt: str
    response: str = ""


def make_messages(b: Blackboard) -> list[JSON]:
    return [{"role": "user", "content": b.prompt}]


def write_response(b: Blackboard, data: str) -> None:
    b.response = data


tree = (
    Tree[Blackboard]("HelloWorld")
    .Sequence()
    ._().LLM("openrouter/openai/gpt-4.1-mini", make_messages)
    ._().WriteBlackboard(write_response)
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
- LiteLLM (LLM calls)
- Redis (required by `Terminable` and `RedisCacher` nodes)
- Uvicorn (required for the HTTP trace server)

## Features

- Minimal, expressive tree builder API
- Async-first execution model
- Leaf / Composite / Decorator nodes built-in
- LLM integration via LiteLLM
- Redis-backed caching and termination signaling
- Trace collection and optional trace storage
- UI trace viewer with HTTP server

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

## Table of Contents <span id="ref"></span>

- [Node Reference](#node-reference)
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
- [License](#license)

## Node Reference <a href="#ref">[↑]</a>

### Leaf Nodes <a href="#ref">[↑]</a>

#### Function <a href="#ref">[↑]</a>

Runs a sync/async function. Returns `OK(data)` for non-`Result` return values, or passes through a `Result`.

```python
tree = (
    Tree()
    .Sequence()
    ._().Function(lambda: "ok")
    .End()
)
```

#### Log <a href="#ref">[↑]</a>

Logs a message into the trace. Always returns `OK(None)`.

```python
tree = (
    Tree()
    .Log("hello")
    .End()
)
```

#### TODO <a href="#ref">[↑]</a>

A placeholder node that always returns `OK(None)`.

```python
tree = (
    Tree()
    .TODO()
    .End()
)
```

#### ShowBlackboard <a href="#ref">[↑]</a>

Logs the current blackboard into the trace and returns `OK(None)`.

```python
tree = (
    Tree()
    .ShowBlackboard()
    .End()
)
```

#### WriteBlackboard <a href="#ref">[↑]</a>

Writes the previous node’s result into the blackboard, and returns `OK(data)`.

```python
tree = (
    Tree()
    .Sequence()
    ._().Function(lambda: 123)
    ._().WriteBlackboard("value")
    .End()
)
```

#### Assert <a href="#ref">[↑]</a>

Checks a boolean condition and returns `OK(True)` or `FAIL(False)`.

```python
tree = (
    Tree()
    .Sequence()
    ._().Assert(lambda: True)
    .End()
)
```

#### Failure <a href="#ref">[↑]</a>

Always returns `FAIL(None)`.

```python
tree = (
    Tree()
    .Failure()
    .End()
)
```

#### Subtree <a href="#ref">[↑]</a>

Runs another tree, optionally with a custom blackboard factory.

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
    ._().Subtree(sub)
    .End()
)
```

#### ParseJSON <a href="#ref">[↑]</a>

Parses JSON from the last result or from a blackboard source, and returns the parsed object.

```python
tree = (
    Tree()
    .Sequence()
    ._().Function(lambda: '{"a":1}')
    ._().ParseJSON(dst="data")
    .End()
)
```

#### LLM <a href="#ref">[↑]</a>

Calls an LLM via LiteLLM and returns the output text. Supports streaming and API key factories.

```python
tree = (
    Tree()
    .Sequence()
    ._().LLM(
        "openrouter/openai/gpt-4.1-mini",
        [{"role": "user", "content": "hi"}],
    )
    .End()
)
```

### Composite Nodes <a href="#ref">[↑]</a>

#### Sequence <a href="#ref">[↑]</a>

Runs children in order. Returns `FAIL` on first failure, otherwise `OK(last_child_data)`.

```python
tree = (
    Tree()
    .Sequence()
    ._().Function(A)
    ._().Function(B)
    .End()
)
```

#### Selector <a href="#ref">[↑]</a>

Runs children in order until one succeeds. Returns the first `OK`, else `FAIL`.

```python
tree = (
    Tree()
    .Selector()
    ._().Failure()
    ._().Function(lambda: "ok")
    .End()
)
```

#### Parallel <a href="#ref">[↑]</a>

Runs children concurrently. Returns `OK` only if all children succeed.

```python
tree = (
    Tree()
    .Parallel(concurrency_limit=2)
    ._().Function(A)
    ._().Function(B)
    .End()
)
```

#### Gather <a href="#ref">[↑]</a>

Runs multiple subtrees with their own blackboards and returns a list of results.

```python
tree = (
    Tree()
    .Gather(lambda b: (trees, blackboards))
    .End()
)
```

#### RandomSelector <a href="#ref">[↑]</a>

Randomizes the child order (optionally weighted) and returns the first `OK`.

```python
tree = (
    Tree()
    .RandomSelector(weights=[1, 2, 3])
    ._().Function(A)
    ._().Function(B)
    ._().Function(C)
    .End()
)
```

#### If / Else <a href="#ref">[↑]</a>

Conditional branch. If the condition is false and no else branch exists, returns `OK(None)`.

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

### Decorator Nodes <a href="#ref">[↑]</a>

#### ForceOk <a href="#ref">[↑]</a>

Forces the result status to `OK`, optionally with a custom data factory.

```python
tree = (
    Tree()
    .ForceOk()
    ._().Failure()
    .End()
)
```

#### ForceFail <a href="#ref">[↑]</a>

Forces the result status to `FAIL`, optionally with a custom data factory.

```python
tree = (
    Tree()
    .ForceFail()
    ._().Function(lambda: "x")
    .End()
)
```

#### Return <a href="#ref">[↑]</a>

Preserves child status but replaces data with a factory result.

```python
tree = (
    Tree()
    .Return(lambda b: "data")
    ._().Function(A)
    .End()
)
```

#### Invert <a href="#ref">[↑]</a>

Inverts child status while keeping data.

```python
tree = (
    Tree()
    .Invert()
    ._().Failure()
    .End()
)
```

#### Retry <a href="#ref">[↑]</a>

Retries a child on failure for up to `max_tries` with optional sleeps.

```python
tree = (
    Tree()
    .Retry(max_tries=3, sleep_secs=0.1)
    ._().Function(A)
    .End()
)
```

#### While <a href="#ref">[↑]</a>

Repeats child while condition is true, returns the last successful result.

```python
tree = (
    Tree()
    .While(lambda b: b.count < 3)
    ._().Function(step)
    .End()
)
```

#### Timeout / Fallback <a href="#ref">[↑]</a>

Runs a child with a time limit. On timeout, runs the fallback child if provided.

```python
tree = (
    Tree()
    .Timeout(1.0)
    ._().Function(slow)
    ._().Function(on_timeout)
    .End()
)
```

#### RedisCacher <a href="#ref">[↑]</a>

Caches child results in Redis. Optional `value_validator` invalidates stale cache.

```python
tree = (
    Tree()
    .RedisCacher(redis_client, key_func=lambda b: "k")
    ._().Function(A)
    .End()
)
```

#### Terminable <a href="#ref">[↑]</a>

Runs a child while polling a Redis key for termination. Optionally runs a fallback.

```python
tree = (
    Tree()
    .Terminable(lambda b: f"stop:{b.job_id}")
    ._().Function(A)
    ._().Fallback()
    ._()._().Function(B)
    .End()
)
```

#### Wrapper <a href="#ref">[↑]</a>

Wraps a child with a custom async context manager.

```python
tree = (
    Tree()
    .Wrapper(my_async_cm)
    ._().Function(A)
    .End()
)
```

## Core APIs (Non-Node)

- `Context`: runtime state (blackboard stack, trace root, path)
- `TraceRoot` / `TraceNode`: structured trace tree
- `TraceStorageHandler` / `FileTraceStorageHandler`: save and load traces
- `register_global_hook_after_spawned_task_finish(hook)`: hook for Parallel/Gather/Terminable tasks
- `set_default_llm_api_key_factory(factory_or_key)`: default LLM API key or factory
- `set_default_global_redis_client(url, **kwargs)`: global Redis client for Redis nodes
- `run_httpserver(host, port, trace_dir)` / `create_http_app(...)`: HTTP trace server

## License

MIT. See `LICENSE.txt`.

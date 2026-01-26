"""
tinytasktree
============

A tiny async task-tree orchestrator library for Python, behavior-tree inspired and LLM-ready.
Requires Python 3.13 or 3.14 (3.15 is not yet supported due to upstream PyO3/fastuuid compatibility).

Example::

    @dataclass
    class Blackboard:
        prompt: str
        response: str = ""

    def write_response(b: Blackboard, data: str) -> None:
        b.response = data

    def make_messages(b: Blackboard) -> list[JSON]:
        return [{"role": "user", "content": b.prompt}]

    def on_delta(b: Blackboard, fulltext: str, delta: str, finished: bool) -> None:
        print(delta, end="")

    tree = (
        Tree[Blackboard]("HelloWorld")
        .Sequence()
        ._().LLM("openrouter/google/gemma-3-27b-it:free", make_messages, stream=True, stream_on_delta=on_delta)
        ._().WriteBlackboard(write_response)
        .End()
    )

    async def main():
        context = Context()
        blackboard = Blackboard(prompt="Write a poem of 100 words!")
        async with context.using_blackboard(blackboard):
            result = await tree(context)

        print("Result:", result)
        print("Full Response:", blackboard.response)


### Node Quick Overview (AST)

Leaf Nodes:
    * .Function()
    * .Log()
    * .TODO()
    * .ShowBlackboard()
    * .WriteBlackboard()
    * .Assert()
    * .Failure()
    * .Subtree()
    * .ParseJSON()
    * .LLM()
Decorator Nodes:
    * .ForceOk()
    * .ForceFail()
    * .Return()
    * .Invert()
    * .Retry()
    * .While()
    * .Timeout() / .Fallback()
    * .RedisCacher()
    * .Terminable()
    * .Wrapper()
Composite Nodes:
    * .Sequence()
    * .Selector()
    * .Parallel()
    * .Gather()
    * .RandomSelector()
    * .If() / .Else()

### Trace Supports

* Trace UI Viewer (Also As A Tree)
* Optional trace storage via `TraceStorageHandler` (e.g., `FileTraceStorageHandler` saves JSON to disk)

Example::

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    trace_json = await storage.query(trace_id)

Note: It is safe to reuse a global `FileTraceStorageHandler` instance across runs.

### UI Server

Trace UI (React/Vite) runs separately and proxies to the HTTP server:

    # 1) start backend
    python -m tinytasktree --httpserver --host 127.0.0.1 --port 8000 --trace-dir .traces
    # 2) start UI dev server
    cd ui && npm run dev
    # 3) open the UI
    http://127.0.0.1:5173

### Others

Able to extend builder functions.

### Hooks

Set a global hook to run after any spawned task finishes (success or failure),
such as tasks created by Parallel, Gather, or Terminable:

    def cleanup(context: Context, tracer: Tracer, result: Result) -> None:
        DBSession.close()

    register_global_hook_after_spawned_task_finish(cleanup)

### Threading note:

Recommended: define trees at module scope (built once at import), then reuse; if you build at runtime,
build and run within the same thread.

### Environments

* DISABLE_TASKTREE_LOGGING=1 disables internal tasktree logging.
* DISABLE_LITE_LLM_LOGGING=1 disables LiteLLM logging (sets suppress_debug_info and set_verbose).

### FAQ

Q: I see "coroutine 'close_litellm_async_clients' was never awaited" warnings. Is it a bug?
A: This comes from LiteLLM async client cleanup on process exit. If it bothers you,
   call `await litellm.close_litellm_async_clients()` before exiting your program.

### License

MIT, (c) OrionArm.AI <https://orionarm.ai>.
Author: chao.wang [at] orionarm.ai
"""

import asyncio
import functools
import inspect
import logging
import os
import pickle
import random
import reprlib
import threading
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, is_dataclass
from datetime import date, datetime, timedelta
from enum import Enum, IntEnum
from typing import (
    Any,
    AsyncContextManager,
    AsyncGenerator,
    Awaitable,
    Callable,
    Literal,
    Protocol,
    Self,
    Type,
    TypeVar,
    cast,
    final,
    override,
)

import json_repair
import litellm
import orjson
import redis.asyncio as async_redis

_DISABLE_LITE_LLM_LOGGING = os.getenv("DISABLE_LITE_LLM_LOGGING", "").strip().lower() not in {"", "0", "false"}
if _DISABLE_LITE_LLM_LOGGING:
    litellm.suppress_debug_info = True
    litellm.set_verbose = False

__all__ = (
    "AnyB",
    "B",
    "Context",
    "JSON",
    "JSONLoader",
    "Result",
    "Status",
    "TasktreeError",
    "TasktreeProgrammingError",
    "Tracer",
    "TraceLevel",
    "TraceNode",
    "TraceRoot",
    "TraceStorageHandler",
    "FileTraceStorageHandler",
    "create_http_app",
    "run_httpserver",
    "set_default_llm_api_key_factory",
    "register_global_hook_after_spawned_task_finish",
    "Tree",
)


##############
# Common Types
##############

type JSON = dict[str, Any]


class TasktreeError(Exception): ...


class TasktreeProgrammingError(TasktreeError): ...


class Status(IntEnum):
    """Node execution result's status."""

    OK = 0
    FAIL = 1

    def __str__(self) -> str:
        return "OK" if self == Status.OK else "FAIL"

    def invert(self) -> "Status":
        return Status.FAIL if self == Status.OK else Status.OK


@dataclass
class Result:
    """Node execution result."""

    status: Status
    data: Any = None

    @classmethod
    def OK(cls, data: Any = None) -> "Result":
        return Result(Status.OK, data)

    @classmethod
    def FAIL(cls, data: Any = None) -> "Result":
        return Result(Status.FAIL, data)

    def is_ok(self) -> bool:
        return self.status == Status.OK

    def __repr__(self) -> str:
        return f"{self.status}({reprlib.repr(self.data)})"

    def json(self) -> JSON:
        return {
            "status": str(self.status),
            "data": _try_to_string(self.data) if self.data is not None else None,
        }


# Any type var
T = TypeVar("T")

# The Blackboard is a shared memory structure used for data and state sharing between nodes, usually implemented as a dataclass.
# Ref: https://en.wikipedia.org/wiki/Blackboard_(design_pattern)
# Any blackboard types
type AnyB = Any

# Blackboard Type Vars
B = TypeVar("B")
B1 = TypeVar("B1", default=B)

##############
# Tracer
##############

type TraceLevel = Literal["info", "warning", "error"]


@dataclass
class TraceNode:
    """One TraceNode for one tree Node."""

    name: str = ""
    kind: str = ""
    start_at: datetime = field(default_factory=lambda: datetime.fromtimestamp(0))
    end_at: datetime = field(default_factory=lambda: datetime.fromtimestamp(0))
    duration: timedelta = field(default_factory=timedelta)
    finished: bool = False
    cost: float = 0  # 'dollar' in general
    logs: list[str] = field(default_factory=list)
    result: Result | None = None
    attributes: JSON = field(default_factory=dict)  # attribute.name => attribute.value
    children: dict[str, "TraceNode"] = field(default_factory=dict)  # child.name => child

    ##### privates ####

    def _ensure_child(self, name: str) -> "TraceNode":
        if name not in self.children:
            self.children[name] = TraceNode(name=name)
        return self.children[name]

    def _ensure_path(self, path: list[str]) -> "TraceNode":
        # Always assuming path[0] is current TraceNode.
        current = self
        for name in path[1:]:
            current = current._ensure_child(name)
        return current

    ##### public ####

    def json(self) -> JSON:
        return {
            "name": self.name,
            "kind": self.kind,
            "start_at": self.start_at.isoformat(),
            "end_at": self.end_at.isoformat(),
            "duration": self.duration.total_seconds() * (10**3),  # milliseconds
            "finished": self.finished,
            "cost": self.cost,
            "logs": self.logs,
            "result": self.result.json() if self.result else None,
            # attributes: {k => v(str)}
            "attributes": {k: _try_to_string(v) for k, v in self.attributes.items()},
            "children": {k: v.json() for k, v in self.children.items()},
        }

    def set_kind(self, kind: str) -> None:
        self.kind = kind

    def set_start(self) -> None:
        self.start_at = datetime.now()
        self.finished = False

    def set_end(self, result: Result) -> None:
        self.finished = True
        self.end_at = datetime.now()
        self.duration = self.end_at - self.start_at
        self.result = result

    def incr_cost(self, delta: float) -> None:
        self.cost += delta

    def update_attributes(self, **kwargs) -> None:
        self.attributes.update(kwargs)

    def log(self, msg: str, level: TraceLevel = "info") -> None:
        self.logs.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f" [{level}] : " + msg)

    def error(self, e: str | BaseException) -> None:
        if isinstance(e, str):
            self.log(e, level="error")
            return
        self.log(_format_exception(e), level="error")

    def total_cost(self) -> float:
        return self.cost + sum([ch.total_cost() for ch in self.children.values()])

    def _node_tokens(self) -> dict[str, int] | None:
        tokens_value = self.attributes.get("tokens")
        tokens: dict[str, int] = {}

        if isinstance(tokens_value, dict):
            prompt = _as_int(tokens_value.get("prompt"))
            completion = _as_int(tokens_value.get("completion"))
            total = _as_int(tokens_value.get("total"))
            if total is None and prompt is not None and completion is not None:
                total = prompt + completion
            _merge_token_fields(tokens, prompt, completion, total)
        elif isinstance(tokens_value, str) and tokens_value:
            try:
                parsed = orjson.loads(tokens_value)
                if isinstance(parsed, dict):
                    prompt = _as_int(parsed.get("prompt"))
                    completion = _as_int(parsed.get("completion"))
                    total = _as_int(parsed.get("total"))
                    if total is None and prompt is not None and completion is not None:
                        total = prompt + completion
                    _merge_token_fields(tokens, prompt, completion, total)
            except Exception:
                pass

        if not tokens:
            prompt = _as_int(self.attributes.get("prompt_tokens") or self.attributes.get("prompt"))
            completion = _as_int(self.attributes.get("completion_tokens") or self.attributes.get("completion"))
            total = _as_int(self.attributes.get("total_tokens") or self.attributes.get("total"))
            if total is None and prompt is not None and completion is not None:
                total = prompt + completion
            _merge_token_fields(tokens, prompt, completion, total)

        return tokens or None

    def total_tokens(self) -> dict[str, int]:
        total: dict[str, int] = {}
        _add_token_totals(total, self._node_tokens())
        for child in self.children.values():
            _add_token_totals(total, child.total_tokens())
        return total


@dataclass
class TraceRoot(TraceNode):
    total_duration: timedelta = field(default_factory=lambda: timedelta(seconds=0))

    def json(self) -> JSON:
        d = super().json()
        total_tokens = self.total_tokens()
        d.update(
            {
                "total_cost": self.total_cost(),
            }
        )
        if total_tokens:
            d["total_tokens"] = total_tokens
        return d


type Tracer = TraceNode


class TraceStorageHandler(Protocol):
    async def save(self, trace_root: TraceRoot) -> str:
        """Persist the trace and return a trace_id."""

    async def query(self, trace_id: str) -> JSON:
        """Load a trace by id and return its JSON content."""


class FileTraceStorageHandler:
    def __init__(self, dirpath: str = ".traces") -> None:
        self._dirpath = dirpath

    def _path_for(self, trace_id: str) -> str:
        return os.path.join(self._dirpath, f"{trace_id}.json")

    async def save(self, trace_root: TraceRoot) -> str:
        trace_id = uuid.uuid4().hex
        path = self._path_for(trace_id)
        data = orjson.dumps(trace_root.json(), option=orjson.OPT_INDENT_2)
        await asyncio.to_thread(self._write_file, path, data)
        return trace_id

    async def query(self, trace_id: str) -> JSON:
        path = self._path_for(trace_id)
        data = await asyncio.to_thread(self._read_file, path)
        return cast(JSON, orjson.loads(data))

    def _write_file(self, path: str, data: bytes) -> None:
        os.makedirs(self._dirpath, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def _read_file(self, path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()


##############
# Metrics
##############


class MetricsHandler(Protocol):
    pass


#############
# Context
#############


class Context:
    """Context stores runtime state (e.g., blackboard stack, node path, trace root).

    Propagation strategy:
    1). New asyncio coroutine: Copy context to avoid data races.
    2). Same coroutine (child nodes): Reuse the instance and maintain stacks manually.
    """

    def __init__(
        self,
        *,
        trace_root: TraceRoot | None = None,
        path: list[str] | None = None,
        blackboard_stack: list[AnyB] | None = None,
        last_result: Result | None = None,
        enable_python_logging: bool = True,
        python_logging_indentifier_name: str = "",
        python_logging_indentifier_value: str = "",
    ) -> None:
        self._trace_root = trace_root or TraceRoot()
        self._path: list[str] = path or []
        self._blackboard_stack: list[AnyB] = blackboard_stack or []
        self._last_result = last_result
        self.enable_python_logging = enable_python_logging
        self.python_logging_indentifier_name = python_logging_indentifier_name
        self.python_logging_indentifier_value = python_logging_indentifier_value

    ##### privates #####

    def _current_blackboard(self) -> AnyB:
        if not self._blackboard_stack:
            raise TasktreeProgrammingError("No blackboard!")
        return self._blackboard_stack[-1]

    def _spawn_forward(self, child: str, b: AnyB | None = None) -> "Context":
        """Copy current Context for a new coroutine to run a child node."""
        return Context(
            trace_root=self._trace_root,  # reference
            path=self._path + [child],  # copy
            blackboard_stack=self._blackboard_stack + [b] if b is not None else self._blackboard_stack.copy(),  # copy
            last_result=self._last_result,
            enable_python_logging=self.enable_python_logging,
            python_logging_indentifier_name=self.python_logging_indentifier_name,
            python_logging_indentifier_value=self.python_logging_indentifier_value,
        )

    @asynccontextmanager
    async def _forward(self, child: str, b: AnyB | None = None) -> AsyncGenerator[None, None]:
        """Advance in the same coroutine without copying current Context."""
        try:
            self._path.append(child)
            if b is not None:
                self._blackboard_stack.append(b)
            yield None
        finally:
            self._path.pop(-1)
            if b is not None:
                self._blackboard_stack.pop(-1)

    ##### public #####

    def trace_root(self) -> TraceRoot:
        """Returns the root tracer node."""
        return self._trace_root

    def current_tracer(self) -> Tracer:
        """Return the tracer for the current node."""
        return self._trace_root._ensure_path(self._path)

    def current_path(self) -> list[str]:
        """Return the current node path."""
        return self._path

    def current_blackboard[T](self, class_: Type[T]) -> T:
        """Return the blackboard for the current node.
        Example::

            b = context.current_blackboard(MyBlackboard)
            # b type: MyBlackboard
        """
        if not self._blackboard_stack:
            raise TasktreeProgrammingError("No blackboard!")
        return cast(T, self._blackboard_stack[-1])

    @asynccontextmanager
    async def using_blackboard(self, b: AnyB) -> AsyncGenerator[None, None]:
        try:
            self._blackboard_stack.append(b)
            yield None
        finally:
            self._blackboard_stack.pop(-1)


##################################
# Node Classes (Inheritance Chain)
##################################


class Node[B](ABC):
    """Abstract base of any Node classes."""

    KIND: str = "Node"

    def __init__(self, name: str) -> None:
        self.name = name
        self._parent: Node[B] | None = None
        if self.name:
            self.fullname = f"{self.KIND}({self.name})"
        else:
            self.fullname = self.KIND

    def __repr__(self) -> str:
        return f"<{self.fullname}>"

    async def __call__(self, context: Context) -> Result:
        return await self._call(context, swallow_cancel=True)

    async def _call(self, context: Context, *, swallow_cancel: bool) -> Result:
        """Internal execution helper.

        Default behavior (swallow_cancel=True):
        - Catch CancelledError.
        - Record trace.
        - Return FAIL(None) instead of propagating cancellation.

        Timeout behavior (swallow_cancel=False):
        - Let CancelledError bubble up.
        - Allows asyncio.timeout to raise TimeoutError.
        - Enables Timeout fallback to run.
        """
        tracer = context.current_tracer()
        tracer.set_kind(self.KIND)
        exc: BaseException | None = None
        result: Result | None = None
        try:
            tracer.set_start()
            result = await self._impl(context, tracer)
            tracer.set_end(result)
        except asyncio.CancelledError as e:
            exc = e
            tracer.error(e)
            result = Result.FAIL(None)
            tracer.set_end(result)
            if not swallow_cancel:
                raise
        except Exception as e:
            exc = e
            tracer.error(e)
            result = Result.FAIL(None)
            tracer.set_end(result)
        finally:
            if context.enable_python_logging:
                if result is None:
                    result = Result.FAIL(None)
                identifier = ""
                if context.python_logging_indentifier_name:
                    identifier = " = ".join(
                        [context.python_logging_indentifier_name, context.python_logging_indentifier_value]
                    )
                excstr = _format_exception(exc) if exc else ""
                current_node_duration_str = f"{tracer.duration.total_seconds() * 1000.0:.2f}ms"
                total_duration = datetime.now() - context._trace_root.start_at
                total_duration_str = f"duration({total_duration.total_seconds() * 1000.0:.2f}ms)"
                pathstr = "[ " + " > ".join(context.current_path()) + " ]"
                msg = " ".join(
                    [identifier, total_duration_str, pathstr, "::", str(result), excstr, current_node_duration_str]
                )
                if result.is_ok():
                    logger.info(msg)
                else:
                    logger.error(msg)
        context._last_result = result
        return result

    def OnBuildEnd(self) -> None:
        """
        A callback executed when node construction finishes.
        Used for validation or post-processing actions.
        """
        if inspect.isabstract(self.__class__):
            raise TasktreeProgrammingError(f"Un-Implemented abstract node class: {self.__class__}")

    @abstractmethod
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        """A complete, concrete Node class must implement this method."""
        raise NotImplementedError()


class LeafNode[B](Node[B]):
    """LeafNode has no child."""

    def __init__(self, name: str = "") -> None:
        Node.__init__(self, name)


class InternalNode[B](Node[B]):
    """InternalNode has one or more children."""

    def __init__(self, name: str = "") -> None:
        Node.__init__(self, name)

    @abstractmethod
    def append_child(self, child: Node[B]) -> None:
        raise NotImplementedError()


class SingleChildNode[B](InternalNode[B]):
    """SingleChildNode has exactly a single child."""

    def __init__(self, child: Node[B] | None = None, name: str = "") -> None:
        InternalNode.__init__(self, name)
        self._child = child

    @final
    def child(self) -> Node[B]:
        assert self._child, TasktreeProgrammingError(f"{self}: no child")
        return self._child

    def OnBuildEnd(self) -> None:
        InternalNode.OnBuildEnd(self)
        if self._child is None:
            raise TasktreeProgrammingError(f"Build Error: {self} no child")

    @final
    @override
    def append_child(self, child: Node[B]) -> None:
        self._child = child
        child._parent = self


class DecoratorNode[B](Node[B]):
    """Base class of all decorators.
    A DecoratorNode may have one or two children.
    """

    def __init__(self, name: str = "") -> None:
        Node.__init__(self, name)


class CompositeNode[B](InternalNode[B]):
    """CompositeNode may have multiple children."""

    def __init__(self, children: list[Node[B]] | None = None, name: str = ""):
        InternalNode.__init__(self, name)
        self._children: list[Node[B]] = []

        # avoid children name duplication
        self._rewrited_children_name_list: list[str] = []
        self._rewrited_children_name_set: set[str] = set()

        if children:
            for child in children:
                self.append_child(child)

    def _find_unique_child_name(self, child_name: str) -> str:
        for i in range(1000):
            name = child_name if i == 0 else f"{child_name}_{i}"
            if name not in self._rewrited_children_name_set:
                return name
        return child_name + f"_{random.randint(10000, 20000)}"

    @final
    def children(self) -> list[Node[B]]:
        return self._children

    @final
    def get_unique_child_name(self, index: int) -> str:
        return self._rewrited_children_name_list[index]

    @final
    @override
    def append_child(self, child: Node[B]):
        unique_name = self._find_unique_child_name(child.fullname)
        self._rewrited_children_name_set.add(unique_name)
        self._rewrited_children_name_list.append(unique_name)
        self._children.append(child)
        child._parent = self

    def OnBuildEnd(self) -> None:
        InternalNode.OnBuildEnd(self)
        if not self._children:
            logger.warning(f"Warning: {self}: Empty children")


###########################
# LeafNodes ::BuiltIn Impls
###########################


type Func01 = Callable[[], Any]  # function() -> any
type Func02 = Callable[[], Result]  # function() -> result
type Func03 = Callable[[], Awaitable[Any]]  # async function() -> any
type Func04 = Callable[[], Awaitable[Result]]  # async function() -> result
type Func05[B] = Callable[[B], Any]  # function(blackboard) -> any
type Func06[B] = Callable[[B], Result]  # function(blackboard) -> result
type Func07[B] = Callable[[B], Awaitable[Any]]  # async function(blackboard) -> any
type Func08[B] = Callable[[B], Awaitable[Result]]  # async function(blackboard) -> result
type Func09[B] = Callable[[B, Tracer], Any]  # function(blackboard, tracer) -> any
type Func10[B] = Callable[[B, Tracer], Result]  # function(blackboard, tracer) -> result
type Func11[B] = Callable[[B, Tracer], Awaitable[Any]]  # async function(blackboard, tracer) -> any
type Func12[B] = Callable[[B, Tracer], Awaitable[Result]]  # async function(blackboard, tracer) -> result
type Func[B] = (
    Func01
    | Func02
    | Func03
    | Func04
    | Func05[B]
    | Func06[B]
    | Func07[B]
    | Func08[B]
    | Func09[B]
    | Func10[B]
    | Func11[B]
    | Func12[B]
)


@final
class FunctionNode[B](LeafNode[B]):
    """A leaf node that just calls a given function."""

    KIND = "Function"

    def __init__(self, func: Func[B], name: str = "") -> None:
        name = name or _normalized_func_name(func)
        LeafNode.__init__(self, name)
        self._func = func
        self._is_async = inspect.iscoroutinefunction(func)
        self._func_param_cnt = _inspect_func_parameters_count(func)

    @override
    def OnBuildEnd(self) -> None:
        LeafNode.OnBuildEnd(self)
        assert self._func_param_cnt in {0, 1, 2}, TasktreeProgrammingError(
            f"{self.fullname}:: invalid function params count"
        )

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        d: Any = None
        tracer = context.current_tracer()
        if self._is_async:
            if self._func_param_cnt == 0:
                d = await cast(Func03 | Func04, self._func)()
            elif self._func_param_cnt == 1:
                b = cast(B, context._current_blackboard())
                d = await cast(Func07[B] | Func08[B], self._func)(b)
            elif self._func_param_cnt == 2:
                b = cast(B, context._current_blackboard())
                d = await cast(Func11[B] | Func12[B], self._func)(b, tracer)
            else:
                raise TasktreeProgrammingError("LeafNode: unsupported function type")
        else:
            if self._func_param_cnt == 0:
                d = cast(Func01 | Func02, self._func)()
            elif self._func_param_cnt == 1:
                b = cast(B, context._current_blackboard())
                d = cast(Func05[B] | Func06[B], self._func)(b)
            elif self._func_param_cnt == 2:
                b = cast(B, context._current_blackboard())
                d = cast(Func09[B] | Func10[B], self._func)(b, tracer)
            else:
                raise TasktreeProgrammingError("LeafNode: unsupported function type")
        if isinstance(d, Result):
            return d
        return Result.OK(d)


type LogMessageFactory[B] = Callable[[B], str]


@final
class LogNode[B](LeafNode[B]):
    KIND = "Log"

    def __init__(self, msg_or_factory: str | LogMessageFactory, level: TraceLevel = "info", name: str = "") -> None:
        if not name:
            if callable(msg_or_factory):
                name = _normalized_func_name(msg_or_factory)
        LeafNode.__init__(self, name)
        self._msg_or_factory = msg_or_factory
        self._level = level
        if self.name:
            self.fullname = f"{self.KIND}({self.name}, {self._level})"
        else:
            self.fullname == f"{self.KIND}({self._level})"

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        b = cast(B, context._current_blackboard())
        msg = self._msg_or_factory(b) if callable(self._msg_or_factory) else self._msg_or_factory
        tracer.log(msg, level=self._level)
        return Result.OK(None)


@final
class TODONode[B](LeafNode[B]):
    KIND = "TODO"

    def __init__(self, name: str = "") -> None:
        LeafNode.__init__(self, name)

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        return Result.OK(None)


@final
class FailureNode[B](LeafNode[B]):
    KIND = "Failure"

    def __init__(self, name: str = "") -> None:
        LeafNode.__init__(self, name)

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        return Result.FAIL(None)


type WriteBlackboardFunction[B] = Callable[[B, Any], None]


@final
class WriteBlackboardNode[B](LeafNode[B]):
    KIND = "WriteBlackboard"

    def __init__(self, attr_or_func: str | WriteBlackboardFunction, name: str = "") -> None:
        if not name:
            if callable(attr_or_func):
                name = _normalized_func_name(attr_or_func)
            else:
                name = f"b.{attr_or_func}"
        LeafNode.__init__(self, name)
        self._attr = attr_or_func if isinstance(attr_or_func, str) else ""
        self._func = attr_or_func if callable(attr_or_func) else None
        assert self._attr or self._func, TasktreeProgrammingError("WriteBlackboard: invalid parameter")

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        b = cast(B, context._current_blackboard())
        if context._last_result:
            data = context._last_result.data
            if self._attr:
                setattr(b, self._attr, data)
            elif self._func:
                self._func(b, data)
            return Result.OK(data)
        tracer.log("no last_result", level="warning")
        return Result.OK(None)


@final
class ShowBlackboard[B](LeafNode[B]):
    KIND = "ShowBlackboard"

    def __init__(self, name: str = "") -> None:
        LeafNode.__init__(self, name)

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        return Result.OK(cast(B, context._current_blackboard()))


type ConditionFunction1 = Callable[[], bool]
type ConditionFunction2 = Callable[[], Awaitable[bool]]
type ConditionFunction3[B] = Callable[[B], bool]
type ConditionFunction4[B] = Callable[[B], Awaitable[bool]]
type ConditionFunction5[B] = Callable[[B, Tracer], bool]
type ConditionFunction6[B] = Callable[[B, Tracer], Awaitable[bool]]

type ConditionFunction = (
    ConditionFunction1
    | ConditionFunction2
    | ConditionFunction3
    | ConditionFunction4
    | ConditionFunction5
    | ConditionFunction6
)


def _check_attr_from_blackboard[B](b: B, attr: str) -> bool:
    return bool(getattr(b, attr))


class _ConditionFunctionHandler_Mixin[B](Node[B]):
    def __init__(self, attr_or_condition_func: str | ConditionFunction) -> None:
        condition: ConditionFunction
        name: str = ""
        if callable(attr_or_condition_func):
            condition = attr_or_condition_func
            name = _normalized_func_name(condition)
        else:
            condition = functools.partial(_check_attr_from_blackboard, attr=attr_or_condition_func)
            name = f"b.{attr_or_condition_func}"

        self._rewrited_name = name
        self._condition = condition
        self._is_condition_async = inspect.iscoroutinefunction(condition)
        self._condition_params_cnt = _inspect_func_parameters_count(condition)

    @override
    def OnBuildEnd(self) -> None:
        Node.OnBuildEnd(self)
        assert self._condition_params_cnt in {0, 1, 2}, TasktreeProgrammingError(
            f"{self.fullname} :: invalid condition params count"
        )

    async def _call_condition(self, context: Context, tracer: Tracer) -> bool:
        if self._is_condition_async:
            if self._condition_params_cnt == 0:
                return await cast(ConditionFunction2, self._condition)()
            elif self._condition_params_cnt == 1:
                b = cast(B, context._current_blackboard())
                return await cast(ConditionFunction4, self._condition)(b)
            elif self._condition_params_cnt == 2:
                b = cast(B, context._current_blackboard())
                return await cast(ConditionFunction6, self._condition)(b, tracer)
        else:
            if self._condition_params_cnt == 0:
                return cast(ConditionFunction1, self._condition)()
            elif self._condition_params_cnt == 1:
                b = cast(B, context._current_blackboard())
                return cast(ConditionFunction3, self._condition)(b)
            elif self._condition_params_cnt == 2:
                b = cast(B, context._current_blackboard())
                return cast(ConditionFunction5, self._condition)(b, tracer)
        raise TasktreeProgrammingError  # wont happen


@final
class AssertionNode[B](_ConditionFunctionHandler_Mixin[B], LeafNode[B]):
    KIND = "Assertion"

    def __init__(self, attr_or_condition_func: str | ConditionFunction, name: str = "") -> None:
        _ConditionFunctionHandler_Mixin.__init__(self, attr_or_condition_func)
        LeafNode.__init__(self, self._rewrited_name)

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        assertion_value = False
        try:
            assertion_value = await self._call_condition(context, tracer)
        except AssertionError:
            assertion_value = False
        if not assertion_value:
            return Result.FAIL(False)
        return Result.OK(True)


type SubtreeBlackboardFactory[B, B1] = Callable[[B], B1]


class SubtreeForwarderNode[B, B1](LeafNode[B]):
    KIND = "Subtree"

    def __init__(
        self,
        subtree: "Tree[B1]",
        subtree_blackboard_factory: SubtreeBlackboardFactory[B, B1] | None = None,
        name: str = "",
    ) -> None:
        LeafNode.__init__(self, name or subtree.name)
        self._subtree = subtree
        self._subtree_blackboard_factory = subtree_blackboard_factory

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        b = context._current_blackboard()
        if self._subtree_blackboard_factory:
            b1 = self._subtree_blackboard_factory(b)
            async with context.using_blackboard(b1):
                # Fold the `Subtree` node and `subtree.Root` node on the path
                # So, we do not advance the path by `_subtree.name`, only using an option blackboard.
                return await self._subtree(context)
        return await self._subtree(context)


type BlackboardAttrGetter[B] = Callable[[B], Any]

type JSONLoader = Callable[[str], JSON]

_JSON_FENCE = "```"
_JSON_FENCE_JSON = "```json"


def json_loader_trying_repair(s: str) -> JSON:
    s = s.strip()
    if s.startswith(_JSON_FENCE_JSON):
        s = s[len(_JSON_FENCE_JSON) :]
    elif s.startswith(_JSON_FENCE):
        s = s[len(_JSON_FENCE) :]
    if s.endswith(_JSON_FENCE):
        s = s[: -len(_JSON_FENCE)]
    try:
        return orjson.loads(s)
    except Exception:
        return cast(JSON, json_repair.loads(s))


class ParseJSON[B](LeafNode[B]):
    KIND = "ParseJSON"

    def __init__(
        self,
        src: str | BlackboardAttrGetter[B] | None = None,
        dst: str | WriteBlackboardFunction[B] | None = None,
        json_loader: JSONLoader | None = None,
        name: str = "",
    ) -> None:
        LeafNode.__init__(self, name)
        self._src = src
        self._dst = dst
        self._json_loader = json_loader or json_loader_trying_repair

    def _get_src_data(self, context: Context) -> str:
        if self._src is None:  # source from last_result
            last_result = context._last_result
            return cast(str, last_result.data if last_result else "{}")
        elif isinstance(self._src, str):  # getattr(b, src)
            b = cast(B, context._current_blackboard())
            return getattr(b, self._src)
        elif callable(self._src):  # function(b)
            b = cast(B, context._current_blackboard())
            func = cast(BlackboardAttrGetter[B], self._src)
            return func(b)
        raise TasktreeProgrammingError("ParseJSON: unspported src")

    def _write_dst(self, context: Context, d: JSON) -> None:
        if self._dst is None:  # No writes
            return
        elif isinstance(self._dst, str):  # setattr
            b = cast(B, context._current_blackboard())
            setattr(b, self._dst, d)
            return
        elif callable(self._dst):  # function(b, d)
            b = cast(B, context._current_blackboard())
            func = cast(WriteBlackboardFunction[B], self._dst)
            func(b, d)
            return
        raise TasktreeProgrammingError("ParseJSON: unspported dst")

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        s = self._get_src_data(context)
        tracer.log(f"using json_loader: {_normalized_func_name(self._json_loader)}")
        d = self._json_loader(s)
        if d is None:
            return Result.FAIL({})
        self._write_dst(context, d)
        return Result.OK(d)


type LLMModelFactory[B] = Callable[[B], str]  # function(blackboard) => model
type LLMMessagesFactory[B] = Callable[[B], list[JSON]]  # function(blackboard) => messages
type LLMStreamFactory[B] = Callable[[B], bool]  # function(blackboard) => stream (bool)

type LLMApiKeyFactory1[B] = Callable[[B], str | None]  # function(blackboard) => api_key | None
type LLMApiKeyFactory2[B] = Callable[[B, str], str | None]  # function(blackboard, model_name) => api_key | None
type LLMApiKeyFactory[B] = LLMApiKeyFactory1[B] | LLMApiKeyFactory2[B]

_DEFAULT_GLOBAL_LLM_API_KEY_FACTORY: str | LLMApiKeyFactory[Any] | None = None
_DEFAULT_GLOBAL_LLM_API_KEY_PARAMS_CNT: int = 0


def set_default_llm_api_key_factory(api_key: str | LLMApiKeyFactory[Any] | None) -> None:
    """Sets a global default API key (or factory) for LLM calls.

    This is useful when you don't want to rely on environment variables and prefer
    providing keys at a global scope.

    Example::
        tinytasktree.set_default_llm_api_key_factory(lambda b, model: b.api_keys[model])
    """
    global _DEFAULT_GLOBAL_LLM_API_KEY_FACTORY, _DEFAULT_GLOBAL_LLM_API_KEY_PARAMS_CNT
    _DEFAULT_GLOBAL_LLM_API_KEY_FACTORY = api_key
    if callable(api_key):
        _DEFAULT_GLOBAL_LLM_API_KEY_PARAMS_CNT = _inspect_func_parameters_count(api_key)
    else:
        _DEFAULT_GLOBAL_LLM_API_KEY_PARAMS_CNT = 0


# [async] function(blackboard, full_output, delta_content, finished[, finish_reason])
# when finished = True, delta_content always be empty str.
type LLMStreamOnChunkCallback1[B] = Callable[[B, str, str, bool], None]
type LLMStreamOnChunkCallback2[B] = Callable[[B, str, str, bool], Awaitable[None]]
type LLMStreamOnChunkCallback3[B] = Callable[[B, str, str, bool, str], None]
type LLMStreamOnChunkCallback4[B] = Callable[[B, str, str, bool, str], Awaitable[None]]
type LLMStreamOnChunkCallback[B] = (
    LLMStreamOnChunkCallback1[B]
    | LLMStreamOnChunkCallback2[B]
    | LLMStreamOnChunkCallback3[B]
    | LLMStreamOnChunkCallback4[B]
)


@final
class LLMNode[B](LeafNode[B]):
    KIND: str = "LLM"

    @staticmethod
    def _extract_tokens(usage: Any | None) -> dict[str, int] | None:
        if not isinstance(usage, dict):
            return None

        prompt = _as_int(usage.get("prompt_tokens"))
        completion = _as_int(usage.get("completion_tokens"))
        total = _as_int(usage.get("total_tokens"))
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion

        tokens: dict[str, int] = {}
        if prompt is not None:
            tokens["prompt"] = prompt
        if completion is not None:
            tokens["completion"] = completion
        if total is not None:
            tokens["total"] = total
        return tokens or None

    @staticmethod
    def _compute_tokens(model_name: str, messages_obj: Any, output_text: str) -> dict[str, int] | None:
        try:
            prompt_tokens = litellm.token_counter(model=model_name, messages=messages_obj)
        except Exception:
            prompt_tokens = None
        try:
            completion_tokens = litellm.token_counter(model=model_name, text=output_text, count_response_tokens=True)
        except Exception:
            completion_tokens = None
        if prompt_tokens is None and completion_tokens is None:
            return None
        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        tokens: dict[str, int] = {}
        if prompt_tokens is not None:
            tokens["prompt"] = int(prompt_tokens)
        if completion_tokens is not None:
            tokens["completion"] = int(completion_tokens)
        if total_tokens is not None:
            tokens["total"] = int(total_tokens)
        return tokens or None

    def _resolve_api_key(self, b: B, model: str) -> str | None:
        api_key_source = self._api_key if self._api_key is not None else _DEFAULT_GLOBAL_LLM_API_KEY_FACTORY
        if callable(api_key_source):
            params_cnt = (
                self._api_key_params_cnt if self._api_key is not None else _DEFAULT_GLOBAL_LLM_API_KEY_PARAMS_CNT
            )
            if params_cnt == 1:
                return cast(LLMApiKeyFactory1[B], api_key_source)(b)
            if params_cnt == 2:
                return cast(LLMApiKeyFactory2[B], api_key_source)(b, model)
            raise TasktreeProgrammingError(f"{self.fullname}: api_key factory params count invalid")
        return api_key_source

    def __init__(
        self,
        model: str | LLMModelFactory[B],
        messages: list[JSON] | LLMMessagesFactory[B],
        stream: bool | LLMStreamFactory[B] = False,
        stream_on_delta: LLMStreamOnChunkCallback[B] | None = None,
        api_key: str | LLMApiKeyFactory[B] | None = None,
        name: str = "",
    ) -> None:
        LeafNode.__init__(self, name)
        self._model = model
        self._messages = messages
        self._stream = stream
        self._stream_on_delta = stream_on_delta
        self._api_key = api_key
        self._is_stream_on_delta_async = False
        self._stream_on_delta_params_cnt = 4
        self._api_key_params_cnt = 0
        if stream_on_delta:
            self._is_stream_on_delta_async = inspect.iscoroutinefunction(stream_on_delta)
            self._stream_on_delta_params_cnt = _inspect_func_parameters_count(stream_on_delta)
        if callable(api_key):
            self._api_key_params_cnt = _inspect_func_parameters_count(api_key)

    def _try_record_cost(
        self,
        *,
        tracer: Tracer,
        model: str,
        response: Any | None = None,
        usage: dict[str, Any] | None = None,
        cost_reported: bool = False,
    ) -> bool:
        if cost_reported:
            return True
        try:
            if response is not None:
                hidden = getattr(response, "_hidden_params", None)
                response_cost = None
                if hidden and "response_cost" in hidden:
                    response_cost = hidden["response_cost"]
                elif isinstance(response, dict):
                    response_cost = response.get("_hidden_params", {}).get("response_cost")
                if response_cost is None:
                    try:
                        response_cost = litellm.completion_cost(completion_response=response)
                    except Exception:
                        response_cost = None
                if response_cost is not None:
                    tracer.incr_cost(float(response_cost))
                    return True
            if usage:
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                if prompt_tokens is not None and completion_tokens is not None:
                    prompt_cost, completion_cost = litellm.cost_per_token(
                        model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
                    )
                    tracer.incr_cost(float(prompt_cost) + float(completion_cost))
                    return True
        except Exception:
            return False
        return False

    @override
    def OnBuildEnd(self) -> None:
        LeafNode.OnBuildEnd(self)
        if self._stream_on_delta:
            assert self._stream_on_delta_params_cnt in {4, 5}, TasktreeProgrammingError(
                f"{self.fullname}: stream callback params count invalid"
            )
        if callable(self._api_key):
            assert self._api_key_params_cnt in {1, 2}, TasktreeProgrammingError(
                f"{self.fullname}: api_key factory params count invalid"
            )
        if self._api_key is None and callable(_DEFAULT_GLOBAL_LLM_API_KEY_FACTORY):
            assert _DEFAULT_GLOBAL_LLM_API_KEY_PARAMS_CNT in {1, 2}, TasktreeProgrammingError(
                f"{self.fullname}: default api_key factory params count invalid"
            )

    async def _call_stream_delta_callback(
        self, b: B, full_output: str, delta_content: str, finished: bool, finish_reason: str
    ) -> None:
        if self._stream_on_delta:
            if self._is_stream_on_delta_async:
                if self._stream_on_delta_params_cnt == 4:
                    func1 = cast(LLMStreamOnChunkCallback2[B], self._stream_on_delta)
                    await func1(b, full_output, delta_content, finished)
                elif self._stream_on_delta_params_cnt == 5:
                    func2 = cast(LLMStreamOnChunkCallback4[B], self._stream_on_delta)
                    await func2(b, full_output, delta_content, finished, finish_reason)
            else:
                if self._stream_on_delta_params_cnt == 4:
                    func3 = cast(LLMStreamOnChunkCallback1[B], self._stream_on_delta)
                    func3(b, full_output, delta_content, finished)
                elif self._stream_on_delta_params_cnt == 5:
                    func4 = cast(LLMStreamOnChunkCallback3[B], self._stream_on_delta)
                    func4(b, full_output, delta_content, finished, finish_reason)

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        b = cast(B, context._current_blackboard())
        model = self._model(b) if callable(self._model) else self._model
        messages = self._messages(b) if callable(self._messages) else self._messages
        stream = self._stream(b) if callable(self._stream) else self._stream
        api_key = self._resolve_api_key(b, model)
        output = ""
        last_tokens: dict[str, int] | None = None

        tracer.update_attributes(model=model)
        tracer.update_attributes(messages=messages)
        tracer.update_attributes(stream=stream)
        if api_key is not None:
            tracer.update_attributes(api_key="***")

        finish_reason: str = ""
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if api_key is not None:
            kwargs["api_key"] = api_key
        response = await litellm.acompletion(**kwargs)

        cost_reported = False

        if stream:
            async for chunk in response:
                choices = chunk.get("choices") or []
                cost_reported = self._try_record_cost(
                    tracer=tracer, model=model, usage=chunk.get("usage"), cost_reported=cost_reported
                )
                tokens = self._extract_tokens(chunk.get("usage"))
                if tokens is not None:
                    last_tokens = tokens
                if choices:
                    delta = choices[0]["delta"]
                    delta_content = delta.get("content") or ""
                    output += delta_content
                    fr = choices[0].get("finish_reason")
                    if fr is not None:
                        finish_reason = fr
                    await self._call_stream_delta_callback(b, output, delta_content, False, finish_reason)
            await self._call_stream_delta_callback(b, output, "", True, finish_reason)
        else:
            output = response["choices"][0]["message"]["content"]
            finish_reason = response["choices"][0].get("finish_reason")
            cost_reported = self._try_record_cost(
                tracer=tracer, model=model, response=response, usage=response.get("usage"), cost_reported=cost_reported
            )
            last_tokens = self._extract_tokens(response.get("usage"))

        tracer.update_attributes(output=output)
        tracer.update_attributes(finish_reason=finish_reason)
        if last_tokens is None:
            last_tokens = self._compute_tokens(model, messages, output)
        if last_tokens is not None:
            tracer.update_attributes(tokens=last_tokens)
            if "prompt" in last_tokens:
                tracer.update_attributes(prompt_tokens=last_tokens["prompt"])
            if "completion" in last_tokens:
                tracer.update_attributes(completion_tokens=last_tokens["completion"])
            if "total" in last_tokens:
                tracer.update_attributes(total_tokens=last_tokens["total"])

        return Result.OK(output)


############################
# CompositeNode
############################


@final
class SequenceNode[B](CompositeNode[B]):
    KIND = "Sequence"

    def __init__(self, children: list[Node[B]] | None = None, name: str = "") -> None:
        CompositeNode.__init__(self, children, name)

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        if not self._children:
            return Result.OK(None)
        last_success_child_data: Any = None
        for i, child in enumerate(self._children):
            async with context._forward(self.get_unique_child_name(i)):
                child_result = await child(context)
            if not child_result.is_ok():
                return Result.FAIL(last_success_child_data)
            last_success_child_data = child_result.data
        return Result.OK(last_success_child_data)


@final
class SelectorNode[B](CompositeNode[B]):
    KIND = "Selector"

    def __init__(self, children: list[Node[B]] | None = None, name: str = "") -> None:
        CompositeNode.__init__(self, children, name)

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        if not self._children:
            return Result.OK(None)
        for i, child in enumerate(self._children):
            async with context._forward(self.get_unique_child_name(i)):
                child_result = await child(context)
            if child_result.is_ok():
                return child_result
        return Result.FAIL(None)


@final
class ParallelNode[B](CompositeNode[B]):
    KIND = "Parallel"

    def __init__(self, children: list[Node[B]] | None = None, concurrency_limit: int = 3, name: str = "") -> None:
        CompositeNode.__init__(self, children, name)
        self._concurrency_limit = concurrency_limit

    @override
    def OnBuildEnd(self) -> None:
        CompositeNode.OnBuildEnd(self)
        if self._concurrency_limit <= 0:
            raise TasktreeProgrammingError(f"{self.fullname}: concurrency_limit <= 0 will make deadlocking")

    async def _child_task(
        self,
        child: Node[B],
        context: Context,
        semaphore: asyncio.Semaphore,
    ) -> Result:
        async with semaphore:
            result = await child(context)
        await _call_spawned_task_finish_hook(context, context.current_tracer(), result)
        return result

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        if not self._children:
            return Result.OK(None)
        tracer.update_attributes(concurrency_limit=self._concurrency_limit)
        semaphore = asyncio.Semaphore(self._concurrency_limit)
        tasks = []
        for i, child in enumerate(self._children):
            child_context = context._spawn_forward(self.get_unique_child_name(i))
            tasks.append(self._child_task(child, child_context, semaphore))
        results: list[Result] = await asyncio.gather(*tasks)
        status = Status.OK if all([r.is_ok() for r in results]) else Status.FAIL
        return Result(status, None)


type RandomWeightsFactory[B] = Callable[[B], list[float]]


@final
class RandomSelectorNode[B](CompositeNode[B]):
    KIND = "RandomSelector"

    def __init__(
        self,
        children: list[Node[B]] | None = None,
        weights: list[float] | RandomWeightsFactory[B] | None = None,
        name: str = "",
    ) -> None:
        CompositeNode.__init__(self, children, name)
        self._weights = weights

    @override
    def OnBuildEnd(self) -> None:
        CompositeNode.OnBuildEnd(self)
        if self._weights is not None:
            if isinstance(self._weights, list):
                if len(self._weights) != len(self.children()):
                    raise TasktreeProgrammingError(f"{self.fullname}: RandomSelector len(weights) != len(children)")

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        if not self._children:
            return Result.OK(None)
        b = cast(B, context._current_blackboard())

        weights: list[float] | None = self._weights(b) if callable(self._weights) else self._weights
        if weights is not None:
            if len(weights) != len(self.children()):
                raise TasktreeProgrammingError(f"{self.fullname}: RandomSelector len(weights) != len(children)")
            if any(w <= 0 for w in weights):
                raise TasktreeProgrammingError(f"{self.fullname}: weights must be positive")

        shuffled = _weighted_shuffle(list(enumerate(self._children)), weights=weights)
        tracer.log("shuffled children order: {}".format([x[0] for x in shuffled]))

        for index, child in shuffled:
            child_name = self.get_unique_child_name(index)
            async with context._forward(child_name):
                child_result = await child(context)
                if child_result.is_ok():
                    return child_result
        return Result.FAIL(None)


# function(blackboard) -> (list[tree1], list[blackboard1])
type GatherParamsFactory[B, B1] = Callable[[B], tuple[list["Tree[B1]"], list[B1]]]


@final
class GatherNode[B, B1](LeafNode[B]):
    KIND = "Gather"

    def __init__(self, params_factory: GatherParamsFactory[B, B1], concurrency_limit: int = 3, name: str = ""):
        LeafNode.__init__(self, name)
        self._params_factory = params_factory
        self._concurrency_limit = concurrency_limit

    @override
    def OnBuildEnd(self) -> None:
        LeafNode.OnBuildEnd(self)
        if self._concurrency_limit <= 0:
            raise TasktreeProgrammingError(f"{self.fullname}: concurrency_limit <= 0 will make deadlocking")

    async def _child_task(
        self,
        child_index: int,
        child: Node[B1],
        context: Context,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, Result]:
        async with semaphore:
            result = await child(context)
        await _call_spawned_task_finish_hook(context, context.current_tracer(), result)
        return child_index, result

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        tracer.update_attributes(suggest_fold_children=True)
        b = cast(B, context._current_blackboard())
        trees, blackboards = self._params_factory(b)
        if len(trees) != len(blackboards):
            raise TasktreeProgrammingError(f"{self.fullname}: number of sub trees and blackboards mismatch")
        semaphore = asyncio.Semaphore(self._concurrency_limit)
        tasks = []
        for index, tree in enumerate(trees):
            tree_blackboard = blackboards[index]
            context1 = context._spawn_forward(f"{index}_" + tree.fullname, tree_blackboard)
            task = self._child_task(index, tree, context1, semaphore)
            tasks.append(task)
        child_results: list[tuple[int, Result]] = await asyncio.gather(*tasks)
        child_results.sort(key=lambda x: x[0])
        data_list = [x[1].data for x in child_results]
        status = Status.OK if all([x[1].is_ok() for x in child_results]) else Status.FAIL
        return Result(status, data_list)


############################
# Decorators ::BuiltIn Impls
############################


class _ForwardingChildNode[B](SingleChildNode[B], DecoratorNode[B]):
    def __init__(self, child: Node[B] | None = None, name: str = "") -> None:
        DecoratorNode.__init__(self, name)
        SingleChildNode.__init__(self, child, name)

    @final
    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        child = self.child()
        async with context._forward(child.fullname):
            return await child(context)


type ResultFactory[B] = Callable[[B], Any]


class _ForceResultStatusBaseNode[B](SingleChildNode[B], DecoratorNode[B]):
    _FORCE_STATUS = Status.OK

    def __init__(self, result_factory: ResultFactory[B] | None = None, name: str = "") -> None:
        if not name:
            if result_factory:
                name = _normalized_func_name(result_factory)
            else:
                name = "None"
        DecoratorNode.__init__(self, name)
        SingleChildNode.__init__(self, None, name)
        self._result_factory = result_factory

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        child = self.child()
        async with context._forward(child.fullname):
            child_result = await child(context)
        if self._result_factory is None:
            return Result(self._FORCE_STATUS, child_result.data)
        b = cast(B, context._current_blackboard())
        data = self._result_factory(b)
        return Result(self._FORCE_STATUS, data)


@final
class ForceOkDecoratorNode[B](_ForceResultStatusBaseNode[B]):
    KIND = "ForceOk"
    _FORCE_STATUS = Status.OK

    def __init__(self, result_factory: ResultFactory[B] | None = None, name: str = "") -> None:
        _ForceResultStatusBaseNode.__init__(self, result_factory, name)


@final
class ForceFailDecoratorNode[B](_ForceResultStatusBaseNode[B]):
    KIND = "ForceFail"
    _FORCE_STATUS = Status.FAIL

    def __init__(self, result_factory: ResultFactory[B] | None = None, name: str = "") -> None:
        _ForceResultStatusBaseNode.__init__(self, result_factory, name)


@final
class InvertDecoratorNode[B](SingleChildNode[B], DecoratorNode[B]):
    KIND = "Invert"

    def __init__(self, name: str = "") -> None:
        DecoratorNode.__init__(self, name)
        SingleChildNode.__init__(self, None, name)

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        child = self.child()
        async with context._forward(child.fullname):
            child_result = await child(context)
            status = child_result.status.invert()
            return Result(status, child_result.data)


@final
class ReturnDecoratorNode[B](SingleChildNode[B], DecoratorNode[B]):
    KIND = "Return"

    def __init__(self, result_factory: ResultFactory, name: str = "") -> None:
        name = name or _normalized_func_name(result_factory)
        DecoratorNode.__init__(self, name)
        SingleChildNode.__init__(self, None, name)
        self._result_factory = result_factory

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        child = self.child()
        async with context._forward(child.fullname):
            child_result = await child(context)
        status = child_result.status
        b = cast(B, context._current_blackboard())
        data = self._result_factory(b)
        return Result(status, data)


@final
class RetryDecoratorNode[B](SingleChildNode[B], DecoratorNode[B]):
    KIND = "Retry"

    def __init__(self, max_tries: int, sleep_secs: float | list[float] | None = None, name: str = ""):
        DecoratorNode.__init__(self, name)
        SingleChildNode.__init__(self, None, name)
        self._max_tries = max_tries
        self._sleep_secs = sleep_secs

    def _determine_sleep_secs(self, tries: int) -> float:
        if self._sleep_secs is not None:
            if isinstance(self._sleep_secs, float):
                return self._sleep_secs
            if isinstance(self._sleep_secs, list):
                return self._sleep_secs[tries] if tries < len(self._sleep_secs) else 0.0
        return 0.0

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        child = self.child()
        for tries in range(self._max_tries):
            async with context._forward(child.fullname):
                result = await child(context)
                if result.is_ok():
                    tracer.log(f"result ok, tries => {tries + 1}")
                    return result
                tracer.log(f"result fail: {result}")
            if self._sleep_secs is not None:
                secs = self._determine_sleep_secs(tries)
                tracer.log(f"sleep => {secs}")
                await asyncio.sleep(secs)
        tracer.log(f"tries => {self._max_tries}")
        return Result.FAIL(None)


@final
class WhileLoopDecoratorNode[B](_ConditionFunctionHandler_Mixin[B], SingleChildNode[B], DecoratorNode[B]):
    KIND = "While"

    def __init__(
        self, attr_or_condition_func: str | ConditionFunction, max_loop_times: int = 1000, name: str = ""
    ) -> None:
        _ConditionFunctionHandler_Mixin.__init__(self, attr_or_condition_func)
        DecoratorNode.__init__(self, self._rewrited_name)
        SingleChildNode.__init__(self, None, self._rewrited_name)
        self._max_loop_times = max_loop_times

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        child = self.child()
        n_times = 0
        result = Result.FAIL(None)

        while True:
            if n_times >= self._max_loop_times:
                tracer.log("exceed max_loop_times, breaking")
                break
            if not await self._call_condition(context, tracer):
                break
            async with context._forward(child.fullname):
                child_result = await child(context)
            if child_result.is_ok():
                result = child_result
            else:
                break
            n_times += 1
        return result


@final
class TimeoutDecoratorNode[B](CompositeNode[B], DecoratorNode[B]):
    KIND = "Timeout"

    def __init__(self, secs: float, name: str = ""):
        DecoratorNode.__init__(self, name)
        CompositeNode.__init__(self, None, name)
        self._secs = secs

    @override
    def OnBuildEnd(self) -> None:
        DecoratorNode.OnBuildEnd(self)
        CompositeNode.OnBuildEnd(self)
        if len(self._children) not in {1, 2}:
            raise TasktreeProgrammingError(f"{self.fullname}: must have 1 or 2 children")

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        children = self.children()
        child0 = children[0]
        child0_name = self.get_unique_child_name(0)
        tracer.log(f"timeout seconds config: {self._secs}")
        try:
            async with asyncio.timeout(self._secs):
                async with context._forward(child0_name):
                    return await child0._call(context, swallow_cancel=False)
        except asyncio.TimeoutError as e:
            tracer.error(f"TimeoutError: {child0_name} {_format_exception(e)}")
            if len(children) == 1:
                tracer.log(f"Timedout! returning initials.. child => {child0_name}")
                return Result.FAIL(None)
            fallback = children[1]
            fallback_name = self.get_unique_child_name(1)
            tracer.log(f"Timedout! fallback to node: {fallback_name}")
            async with context._forward(fallback_name):
                return await fallback(context)


@final
class FallbackNode[B](_ForwardingChildNode[B]):
    KIND = "Fallback"

    def __init__(self, name: str = ""):
        _ForwardingChildNode.__init__(self, None, name)

    @override
    def OnBuildEnd(self) -> None:
        _ForwardingChildNode.OnBuildEnd(self)
        assert self._parent
        if self._parent.KIND not in {"Timeout", "Terminable"}:
            raise TasktreeProgrammingError(f"{self.fullname}: Fallback's parent must be one of [Timeout, Terminable]")


type TerminableRedisKeyFunction[B] = Callable[[B], str]


@final
class TerminableDecoratorNode[B](CompositeNode[B], DecoratorNode[B]):
    KIND = "Terminable"

    def __init__(
        self,
        key_func: TerminableRedisKeyFunction,
        redis_client: async_redis.Redis | None = None,
        monitor_interval_ms: float = 500,  # ms
        name: str = "",
    ):
        DecoratorNode.__init__(self, name)
        CompositeNode.__init__(self, None, name)
        self._key_func = key_func
        self._monitor_interval_ms = monitor_interval_ms
        self._redis_client = redis_client or _DEFAULT_GLOBAL_REDIS_INSTANCE

    @override
    def OnBuildEnd(self) -> None:
        DecoratorNode.OnBuildEnd(self)
        CompositeNode.OnBuildEnd(self)
        if len(self.children()) not in {1, 2}:
            raise TasktreeProgrammingError(f"{self.fullname}: must have 1 or 2 children")
        if not self._redis_client:
            raise TasktreeError(f"{self.fullname}: must provide a redis_client instance")

    async def _monitor_termination_signal(self, context: Context) -> None:
        assert self._redis_client
        b = cast(B, context._current_blackboard())
        k = self._key_func(b)
        # We first clear the key, ensures everything is clean.
        await self._redis_client.delete(k)
        while True:
            if await self._redis_client.exists(k):
                await self._redis_client.delete(k)
                return
            await asyncio.sleep(self._monitor_interval_ms / 1000.0)

    async def _run_child(self, child: Node, child_name: str, context: Context) -> Result:
        async with context._forward(child_name):
            result = await child(context)
        await _call_spawned_task_finish_hook(context, context.current_tracer(), result)
        return result

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        child = self._children[0]
        child_name = self.get_unique_child_name(0)

        task1 = asyncio.create_task(self._run_child(child, child_name, context))
        task2 = asyncio.create_task(self._monitor_termination_signal(context))

        done, pending = await asyncio.wait([task1, task2], return_when=asyncio.FIRST_COMPLETED)
        task1_done = task1 in done

        # Cancel others at first.
        for t in pending:
            t.cancel()

        # Wait remaining cancellations completed.
        await asyncio.wait(pending, return_when=asyncio.ALL_COMPLETED)

        if task1_done:
            return cast(Result, task1.result())

        tracer.log("cancelled")
        if len(self._children) > 1:
            fallback = self._children[1]
            fallback_name = self.get_unique_child_name(1)
            tracer.log("trying fallback")
            async with context._forward(fallback_name):
                return await fallback(context)
        tracer.log("no fallback, returning FAIL(None)")
        return Result.FAIL(None)


type RedisCacherKeyFunction[B] = Callable[[B], str]  # function(blackboard) -> key
#  # function(blackboard [, tracer]) -> value_validator_string
type RedisCacherValueValidator1[B] = Callable[[B], str]
type RedisCacherValueValidator2[B] = Callable[[B, Tracer], str]
type RedisCacherValueValidator[B] = RedisCacherValueValidator1[B] | RedisCacherValueValidator2[B]
# seconds (int | float), timedelta, random in [min_timedelta, max_timedelta]
type Cache_Expiration = int | float | timedelta | tuple[timedelta, timedelta]
# function(blackboard) -> bool
type RedisCacherEnabledFunction[B] = Callable[[B], bool]


@final
class RedisCacherNode[B](SingleChildNode[B], DecoratorNode[B]):
    KIND = "RedisCacher"

    def __init__(
        self,
        key_func: RedisCacherKeyFunction[B],
        redis_client: async_redis.Redis | None = None,
        expiration: Cache_Expiration = timedelta(hours=1),
        value_validator: RedisCacherValueValidator[B] | None = None,
        enabled: bool | RedisCacherEnabledFunction[B] = True,
        name: str = "",
    ) -> None:
        DecoratorNode.__init__(self, name)
        SingleChildNode.__init__(self, None, name)
        self._key_func = key_func
        self._redis_client = redis_client or _DEFAULT_GLOBAL_REDIS_INSTANCE
        self._value_validator = value_validator
        self._value_validator_params_cnt = _inspect_func_parameters_count(value_validator) if value_validator else 0
        self._ex = expiration
        self._enabled = enabled

    @override
    def OnBuildEnd(self) -> None:
        DecoratorNode.OnBuildEnd(self)
        SingleChildNode.OnBuildEnd(self)
        if self._value_validator:
            if self._value_validator_params_cnt not in {1, 2}:
                raise TasktreeProgrammingError(f"{self.fullname}: value_validator params count invalid")
        if not self._redis_client:
            raise TasktreeError(f"{self.fullname}: must provide a redis_client instance")

    def _compute_enabled(self, context: Context) -> bool:
        if callable(self._enabled):  # function(blackboard) -> bool
            b = cast(B, context._current_blackboard())
            func = cast(RedisCacherEnabledFunction[B], self._enabled)
            return func(b)
        return self._enabled  # bool

    def _compute_ex(self) -> timedelta:
        if isinstance(self._ex, timedelta):  # Fixed Timedelta
            return self._ex
        elif isinstance(self._ex, (float, int)):  # Seconds
            return timedelta(seconds=self._ex)
        elif isinstance(self._ex, tuple):  # Random Duration
            ex = cast(tuple[timedelta, timedelta], self._ex)
            min_t, max_t = ex
            min_t_secs, max_t_secs = (
                int(min_t.total_seconds()),
                int(max_t.total_seconds()),
            )
            secs = random.randint(min_t_secs, max_t_secs)
            return timedelta(seconds=secs)
        raise TasktreeProgrammingError("RedisCacher: invalid expiration param")

    def _compute_value_validator(self, context: Context, tracer: Tracer) -> str:
        b = cast(B, context._current_blackboard())
        if self._value_validator_params_cnt == 1:
            return cast(RedisCacherValueValidator1[B], self._value_validator)(b)
        elif self._value_validator_params_cnt == 2:
            return cast(RedisCacherValueValidator2[B], self._value_validator)(b, tracer)
        return ""  # won't happen

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        assert self._redis_client
        enabled = self._compute_enabled(context)
        child = self.child()
        b = cast(B, context._current_blackboard())
        k = self._key_func(b)
        validation = str(self._compute_value_validator(context, tracer)) if self._value_validator else ""

        if enabled:
            w = await self._redis_client.get(k)
            if w:
                try:
                    x = pickle.loads(w)
                    if self._value_validator:  # Need validation
                        if x["validation"] == validation:
                            # Succeeds only if key exists and validation pass.
                            tracer.log(f"cache hit, key: {k}, validation: {validation}")
                            return Result.OK(x["value"])
                        else:
                            tracer.log(
                                "cache key found, but validation value changed: "
                                + x["validation"]
                                + f"(expect: {validation})=> deleting key {k}"
                            )
                            await self._redis_client.delete(k)
                    else:
                        # No validation, directly returns the cached value
                        return Result.OK(x["value"])
                except Exception as e:
                    # (KeyError, pickle.UnpicklingError)
                    tracer.error(f"{e} => miss")
            else:
                tracer.log(f"cache miss, key: {k}")
        else:
            tracer.log("cache disabled")

        async with context._forward(child.fullname):
            result = await child(context)

        if enabled and result.is_ok():
            # Sets cache only if child runs successfully.
            ex = self._compute_ex()
            x1 = {"value": result.data, "validation": validation}
            w = pickle.dumps(x1)
            await self._redis_client.set(k, w, ex=ex)
            tracer.log(f"cache set (on ok), ex: {int(ex.total_seconds())}s, validation: {validation}")
        return result


type WrapperFunction[B] = Callable[[Node[B], Context], AsyncContextManager[Result]]


class WrapperNode[B](SingleChildNode[B], DecoratorNode[B]):
    KIND = "Wrapper"

    def __init__(self, func: WrapperFunction, name: str = "") -> None:
        DecoratorNode.__init__(self, name)
        SingleChildNode.__init__(self, None, name)
        self._func = func

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        child = self.child()
        async with context._forward(child.fullname):
            cm = self._func(child, context)
            if not hasattr(cm, "__aenter__") or not hasattr(cm, "__aexit__"):
                raise TasktreeProgrammingError(
                    f"{self.fullname}: Wrapper function must return an async context manager"
                )
            async with cm as result:
                return result


##############
# If/Else
##############


@final
class IfNode[B](_ConditionFunctionHandler_Mixin[B], CompositeNode[B], DecoratorNode[B]):
    KIND = "If"

    def __init__(self, attr_or_condition_func: str | ConditionFunction, name: str = "") -> None:
        _ConditionFunctionHandler_Mixin.__init__(self, attr_or_condition_func)
        DecoratorNode.__init__(self, self._rewrited_name)
        CompositeNode.__init__(self, None, self._rewrited_name)

    @override
    def OnBuildEnd(self) -> None:
        DecoratorNode.OnBuildEnd(self)
        CompositeNode.OnBuildEnd(self)
        _ConditionFunctionHandler_Mixin.OnBuildEnd(self)
        if len(self._children) not in {1, 2}:
            raise TasktreeProgrammingError(f"{self.fullname}: must have 1 or 2 children")

    @override
    async def _impl(self, context: Context, tracer: Tracer) -> Result:
        child0 = self._children[0]
        child0_name = self.get_unique_child_name(0)

        if await self._call_condition(context, tracer):
            tracer.log("condition: true")
            async with context._forward(child0_name):
                return await child0(context)
        else:
            tracer.log("condition: false")
            if len(self._children) == 1:
                # No else (fallback)
                tracer.log("no fallback(else), returning ok")
                return Result.OK(None)
            tracer.log("executing fallback(else)")
            child1 = self._children[1]
            child1_name = self.get_unique_child_name(1)
            async with context._forward(child1_name):
                return await child1(context)


@final
class ElseNode[B](_ForwardingChildNode[B]):
    KIND = "Else"

    def __init__(self, name: str = ""):
        _ForwardingChildNode.__init__(self, None, name)

    @override
    def OnBuildEnd(self) -> None:
        _ForwardingChildNode.OnBuildEnd(self)
        assert self._parent
        if self._parent.KIND != "If":
            raise TasktreeProgrammingError(f"{self.fullname}: Else must be a child of a node If")


##############
# Tree
##############


class Tree[B](_ForwardingChildNode[B]):
    """
    Tree root node, and builder for constructing tree structures.

    Example::

        tree = (
            Tree("MyTree")
            .Sequence()
            ._().Function(A)
            ._().Function(B)
            .End()
        )

    Recommended: define trees at module scope (built once at import), then reuse; if you build at runtime,
    build and run within the same thread.

    Extending Building Capabilities: To add custom nodes (e.g., a custom `DecoratorNode`),
    subclass `Tree` and use the `_attach` method to integrate your node into the builder chain.

    For lightweight customization without subclassing, use `.Wrapper(...)` to wrap an existing
    subtree with a custom async context manager (e.g., timing, logging, or resource management).

    Example::

        class MyDecoratorNode[B](DecoratorNode[B]):
            @override
            async def _impl(self, context: Context, tracer: Tracer) -> Result:
                ... # implementation

        class MyTree(Tree):
            def MyDecorator(self, ...) -> Self:
                return self._attach(MyDecoratorNode(...))

        tree = (
            MyTree("MyTree")
            .MyDecorator()
            ._().Function(A)
            .End()
        )
    """

    KIND = "Tree"

    def __init__(self, name: str = "") -> None:
        _ForwardingChildNode.__init__(self, None, name)

        self._stack: list[InternalNode[B]] = [self]
        self._level: int = 1

    @final
    @override
    async def __call__(self, context: Context) -> Result:
        try:
            is_calling_from_root = False
            if not context._path:
                is_calling_from_root = True
                # Called as the most root tree
                context._path = ["ROOT", self.fullname]
                context._trace_root.set_start()
                context._trace_root.name = "ROOT"
                context._trace_root.kind = "ROOT"
            result = await _ForwardingChildNode.__call__(self, context)
            context._trace_root.set_end(result)
            return result
        finally:
            if is_calling_from_root:
                # rollback this context
                context._path = []

    #########################
    # Builder
    #########################

    def _stack_pop(self) -> None:
        node = self._stack.pop(-1)
        node.OnBuildEnd()

    def _adjust(self) -> None:
        if self._level > len(self._stack):
            raise TasktreeProgrammingError(f"too much indent: {self._stack[-1]}")
        while self._level < len(self._stack):
            self._stack_pop()

    def _attach_leaf_node(self, node: LeafNode[B]) -> Self:
        self._adjust()
        self._stack[-1].append_child(node)
        self._level = 1
        return self

    def _attach_internal_node(self, node: InternalNode[B]) -> Self:
        self._adjust()
        parent = self._stack[-1]
        self._stack.append(node)
        parent.append_child(node)
        self._level = 1
        return self

    def _attach(self, node: Node[B]) -> Self:
        if isinstance(node, LeafNode):
            self._attach_leaf_node(cast(LeafNode[B], node))
        elif isinstance(node, InternalNode):
            self._attach_internal_node(cast(InternalNode[B], node))
        return self

    def End(self) -> Self:
        """
        Ends the current tree builder.

        Note: This closes the tasktree itself (all open builder state).
        It is not used to close individual blocks.
        """
        while self._stack:
            self._stack_pop()
        return self

    def _(self) -> Self:
        self._level += 1
        return self

    #########################
    # Builder :: Leaf
    #########################

    def Function(self, func: Func[B], name: str = "") -> Self:
        """
        Executes function `func` (name defaults to `func.__name__`).

        Return Handling:
        1) Wraps non-Result returns in `Result.OK(value)`.
        2) Passes `Result` returns through directly.

        Supported Signatures (Sync or Async):
        * f() -> Any | Result
        * f(blackboard) -> Any | Result
        * f(blackboard, tracer) -> Any | Result

        Any exception raised results in `Result.FAIL(None)`.

        Example::

            def rank(b: Blackboard):
                b.data_list.sort()

            Tree()
            .Sequence()
            ._().Function(rank)
            .End()
        """
        return self._attach(FunctionNode[B](func, name))

    def Log(self, msg_or_factory: str | LogMessageFactory, level: TraceLevel = "info", name: str = "") -> Self:
        """
        Logs a message to the Trace.

        :param msg_or_factory: A string message,
            or a message factory function with the form: `function(blackboard) -> str`.
        :param level: The log level to use: "info" | "error".

        This node always returns `Result.OK(None)`.

        Example::

            Tree()
            .Log(lambda b: "debugging input: " + b.input)
            .End()

        """
        return self._attach(LogNode[B](msg_or_factory, level, name))

    def TODO(self, name: str = "") -> Self:
        """
        Adds a TODO item. Useful as a placeholder or for sketching out the tree structure
        during the early stages of development.

        It only simply returns `Result.OK(None)` without any internal operations.

        Example::

            Tree()
            .Sequence()
            ._().TODO("Call LLM")
            ._().TODO("Parse the result to json")
            ._().TODO("Reply to user")
            .End()
        """
        return self._attach(TODONode[B](name))

    def Failure(self, name: str = "") -> Self:
        """
        A leaf node that always fails. Performs no logic and simply returns `Result.FAIL(None)`.

        Typically used to explicitly terminate the current execution flow.

        Example::

            Tree()
            .Selector()
            ._().Function(check)
            ._().Failure()
            .End()
        """
        return self._attach(FailureNode[B](name))

    def ShowBlackboard(self, name: str = "") -> Self:
        """
        Print the blackboard content to trace.
        Always returns `Result.OK(blackboard)`
        """
        return self._attach(ShowBlackboard[B](name))

    def WriteBlackboard(self, attr_or_func: str | WriteBlackboardFunction, name: str = "") -> Self:
        """
        Captures the latest execution result (`last_result`, in current coroutine) and writes it to the blackboard.

        Return Handling:
        - If `last_result` exists, returns `OK(last_result.data)` after writing.
        - If no `last_result` exists, returns `OK(None)` and logs a warning.

        :param attr_or_func: Either an attribute name (string) or a custom setter function.
            - String: Assigns `last_result` to `blackboard.<attr_or_func>`.
            - Function: A function with signature `f(blackboard, last_result) -> None` for manual assignment.

        Example::

            # Case 1: Using a string attribute name
            Tree()
            .Sequence()
            ._().Function(add)
            ._().WriteBlackboard("result")
            .End()

            # Case 2: Using a custom setter function
            def set_result(blackboard: Blackboard, last_result: int):
                blackboard.result = last_result

            Tree()
            .Sequence()
            ._().Function(add)
            ._().WriteBlackboard(set_result)
            .End()
        """
        return self._attach(WriteBlackboardNode[B](attr_or_func, name))

    def Assert(self, attr_or_condition_func: str | ConditionFunction, name: str = "") -> Self:
        """
        Asserts a condition.

        - Status & Data: Returns `OK(True)` if the assertion passes; otherwise returns `FAIL(False)`.

        :param attr_or_condition_func: A string attribute name or a condition function.
            - String: Checks the boolean value of `blackboard.<attr>`.
            - Function: Supports sync or async with signatures:
                - `() -> bool`
                - `(blackboard) -> bool`
                - `(blackboard, tracer) -> bool`

        Note: An `AssertionError` raised during execution is treated as a failure.

        Example::

            Tree()
            .Sequence()
            ._().Assert(lambda b: b.input)  # Equivalent to .Assert("input")
            ._().Function(DoA)
            .End()

        In this example, if the assertion fails, the `Sequence` terminates immediately and `DoA` is skipped.
        """
        return self._attach(AssertionNode[B](attr_or_condition_func, name))

    def ParseJSON(
        self,
        src: str | BlackboardAttrGetter[B] | None = None,
        dst: str | WriteBlackboardFunction[B] | None = None,
        json_loader: JSONLoader | None = None,
        name: str = "",
    ) -> Self:
        """
        Parses a JSON string into a Python object.

        - Status: Returns `OK` if parsing is successful; returns `FAIL({})` if parsing results in `None`.
        - Data: Returns the parsed JSON object.

        :param src: Source of the JSON string.
            - `None`: Uses `last_result.data` from the previous node.
            - `str`: Reads from `blackboard.<src>`.
            - `Callable`: A factory function `f(blackboard) -> str`.
        :param dst: Destination for the parsed object.
            - `None`: Does not write to the blackboard.
            - `str`: Writes to `blackboard.<dst>`.
            - `Callable`: A setter function `f(blackboard, parsed_data) -> None`.
        :param json_loader: A function to parse json into `JSON`, form: `f(str) -> JSON`.
            Defaults to `json_loader_trying_repair`, which strips common ```json fences and
            tries `json_repair` on orjson load failure.
        :param name: Optional node name.

        Example::

            Tree()
            .Sequence()
            ._().LLM(messages=[{"role": "user", "content": "Return a JSON list of colors"}])
            ._().ParseJSON(dst="color_list")
            .End()
        """
        return self._attach(ParseJSON[B](src, dst, json_loader, name))

    def LLM(
        self,
        model: str | LLMModelFactory[B],
        messages: list[JSON] | LLMMessagesFactory[B],
        stream: bool | LLMStreamFactory[B] = False,
        stream_on_delta: LLMStreamOnChunkCallback[B] | None = None,
        api_key: str | LLMApiKeyFactory[B] | None = None,
        name: str = "",
    ) -> Self:
        """
        Invokes an LLM (Large Language Model).

        Based on LiteLLM: https://docs.litellm.ai/docs

        - Status: Returns `OK` upon successful completion.
        - Data: Returns the final output text: `OK(output_text)`.

        :param model: Model name string or a factory function `f(blackboard) -> str`.
        :param messages: A list of message objects or a factory function `f(blackboard) -> list[JSON]`.
        :param stream: Boolean or a factory function `f(blackboard) -> bool` to enable streaming.
        :param stream_on_delta: Optional callback for streaming. Supports sync or async with signatures:
            - `[async] (blackboard, full_text: str, delta_content: str, finished: bool)`
            - `[async] (blackboard, full_text: str, delta_content: str, finished: bool, finish_reason: str)`
            Note: When `finished` is True, `delta_content` is an empty string.
        :param api_key: Optional API key string or a factory function:
            - `f(blackboard) -> str | None`
            - `f(blackboard, model_name) -> str | None`
            Resolution order:
            1) `api_key` passed to this node
            2) `set_default_llm_api_key_factory()`
            3) Environment variables (LiteLLM)
            4) None
        :param name: Optional name for the node.

        Example::

            Tree()
            .Sequence()
            ._().LLM("openrouter/openai/gpt-4.1-mini", [{"role": "user", "content": "Hello!"}], stream=True, stream_on_delta=lambda b, text, delta, finished: print(delta, end=""))
            ._().WriteBlackboard("ai_response")
            .End()
        """
        return self._attach(LLMNode[B](model, messages, stream, stream_on_delta, api_key, name))

    def Subtree[B1](
        self,
        subtree: "Tree[B1]",
        subtree_blackboard_factory: SubtreeBlackboardFactory[B, B1] | None = None,
        name: str = "",
    ) -> Self:
        """
        Attaches a subtree.

        :param subtree: The subtree to be mounted.
        :param subtree_blackboard_factory: Optional factory function `f(blackboard) -> SubBlackboard`.
            Used to derive a specific blackboard for the subtree. If None, the current blackboard is reused.

        Example::

            def construct_subtree_blackboard(b: Blackboard) -> SubtreeBlackboard:
                ...

            Tree()
            .Sequence()
            ._().Subtree(subtree, construct_subtree_blackboard)
            ._().WriteBlackboard("subtree_result")
            .End()
        """
        return self._attach(SubtreeForwarderNode[B, B1](subtree, subtree_blackboard_factory, name))

    #########################################
    # Builder :: Decorators
    #########################################

    def ForceOk(self, result_factory: ResultFactory[B] | None = None, name: str = "") -> Self:
        """
        Forces the child node to return success (`OK`).

        :param result_factory: Optional function `f(blackboard) -> Any`.
            - Default: Returns `Result.OK(child_result.data)`.
            - If provided: Returns `Result.OK(result_factory(blackboard))`.

        Example::

            # Always returns OK with the child's original data
            Tree()
            .ForceOk()
            ._().Function(func1)
            .End()

            # Always returns OK with custom data from the blackboard
            Tree()
            .ForceOk(lambda b: b.result)
            ._().Function(func1)
            .End()
        """
        return self._attach(ForceOkDecoratorNode[B](result_factory, name))

    def ForceFail(self, result_factory: ResultFactory[B] | None = None, name: str = "") -> Self:
        """
        Forces the child node to return success (`FAIL`).

        :param result_factory: Optional function `f(blackboard) -> Any`.
            - Default: Returns `Result.FAIL(child_result.data)`.
            - If provided: Returns `Result.FAIL(result_factory(blackboard))`.

        Example::

            # Always returns FAIL with the child's original data
            Tree()
            .ForceFail()
            ._().Function(func1)
            .End()

            # Always returns FAIL with custom data from the blackboard
            Tree()
            .ForceFail(lambda b: b.result)
            ._().Function(func1)
            .End()
        """
        return self._attach(ForceFailDecoratorNode[B](result_factory, name))

    def Invert(self, name: str = "") -> Self:
        """
        Inverts the child node's result status (Decorator).

        - If the child succeeds: returns `FAIL(child_result.data)`.
        - If the child fails: returns `OK(child_result.data)`.

        Example::

            Tree()
            .Invert()
            ._().Function(A)
            .End()
        """
        return self._attach(InvertDecoratorNode[B](name))

    def Return(self, result_factory: ResultFactory, name: str = "") -> Self:
        """
        Decorator that executes the child subtree and overrides its result data.

        - Status: Inherits the status (OK/FAIL) from the child's result.
        - Data: Returns the data generated by `result_factory(blackboard)`.

        Effectively returns: `Result(child_result.status, data=result_factory(blackboard))`.

        Example::

            Tree()
            .Return(lambda b: b.result)
            ._().Leaf(A())
            .End()
        """
        return self._attach(ReturnDecoratorNode[B](result_factory, name))

    def Retry(self, max_tries: int, sleep_secs: float | list[float] | None = None, name: str = "") -> Self:
        """
        Retry decorator.

        Retries the child node until it succeeds or the maximum number of attempts is reached.

        - Status: Returns `OK` if the child succeeds; returns `FAIL` if all attempts fail.
        - Data: Returns the successful `child_result.data`. Returns `FAIL(None)` if exhausted.

        :param max_tries: Total number of attempts (including the initial execution).
        :param sleep_secs: Delay (in seconds) before the next retry.
            Supports a fixed `float` or a `list[float]` for sequential delays.

        Example::

            Tree()
            .Retry(max_tries=3, sleep_secs=[1.0, 2.0])
            ._().Function(A)
            .End()

        In this example:
        1. Retries up to 3 times.
        2. Returns `OK` immediately if A succeeds.
        3. Returns `FAIL(None)` if A fails on all 3 attempts.
        """
        return self._attach(RetryDecoratorNode[B](max_tries, sleep_secs, name))

    def While(
        self, attr_or_condition_func: str | ConditionFunction, max_loop_times: int = 1000, name: str = ""
    ) -> Self:
        """
        Executes the child node repeatedly as long as a condition remains true.

        - Status & Data: Returns the result of the last successful child execution.
          Returns `FAIL(None)` if the loop never runs or the first execution fails.

        Termination Conditions:
        1. The `condition` evaluates to `False`.
        2. The child node returns `FAIL`. (Use `ForceOk` to continue despite failures).
        3. The `max_loop_times` limit is reached (safety fuse).

        :param condition: The loop condition.
            - String: Checks boolean value of `blackboard.<attr>`.
            - Function: any form of: sync or async with following signatures:
              - `()`
              - `(blackboard)`
              - `(blackboard, tracer) -> bool`
        :param max_loop_times: Maximum number of iterations to prevent infinite loops.

        Example::

            Tree()
            .While(lambda b: not b.finished, max_loop_times=1000)
            ._().Sequence()
            ._()._().Function(A)
            ._()._().WriteBlackboard("finished")
            .End()

        In this example, child A runs repeatedly until `finished` is set to True or the limit is reached.
        """
        return self._attach(WhileLoopDecoratorNode[B](attr_or_condition_func, max_loop_times, name))

    def Timeout(self, secs: float, name: str = "") -> Self:
        """
        Timeout decorator.

        - First child: The target subtree to execute.
        - Second child (optional): The fallback subtree to execute if a timeout occurs.

        Behavior:
        1. If the main task finishes within the time limit, its result is returned.
        2. If the timeout is reached, the main task is interrupted.
        3. After an interruption, it executes the second child (if provided) and returns its result.
        4. If no second child exists, it simply returns `FAIL(None)`.

        Example (With Fallback)::

            Tree()
            .Timeout(3)
            ._().Function(DoA) # Main task
            ._().Fallback()
            ._()._().Function(DoB) # Executes if DoA takes > 3s
            .End()

        Example (Without Fallback)::

            Tree()
            .Timeout(3)
            ._().Function(DoA)
            .End()
            # Returns FAIL(None) if DoA takes > 3s
        """
        return self._attach(TimeoutDecoratorNode[B](secs, name))

    def Fallback(self, name: str = "") -> Self:
        """
        Used in conjunction with the `Timeout` node.
        Refer to the `Timeout` documentation for details.
        """
        return self._attach(FallbackNode[B](name))

    def Wrapper(self, func: WrapperFunction, name: str = "") -> Self:
        """
        Wraps the child execution with a custom async context manager (Decorator).

        :param func: Async context manager `f(child, context) -> AsyncContextManager[Result]`.
            Typically, it calls `await child(context)` and yields the result, but it can
            also return a custom Result.

        Example::

            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def wrap(child, context):
                print("before")
                try:
                    result = await child(context)
                    yield result
                finally:
                    print("after")

            Tree()
            .Wrapper(wrap)
            ._().Function(DoA)
            .End()
        """
        return self._attach(WrapperNode(func, name))

    def RedisCacher(
        self,
        key_func: RedisCacherKeyFunction[B],
        redis_client: async_redis.Redis | None = None,
        expiration: Cache_Expiration = timedelta(hours=1),
        value_validator: RedisCacherValueValidator[B] | None = None,
        enabled: bool | RedisCacherEnabledFunction[B] = True,
        name: str = "",
    ) -> Self:
        """
        Caches the child node's result in Redis (Decorator).
        Available only if module `redis.asyncio` is installed.

        - Status/Data: If a valid cache entry exists, returns `OK(cached_data)`.
          Otherwise, executes the child and stores its result in Redis if it returns `OK`.

        :param key_func: Factory function `f(blackboard) -> str` to generate the cache key.
        :param redis_client: An asynchronous Redis client instance.
            defaults to `_DEFAULT_GLOBAL_REDIS_INSTANCE` (which can be set by `set_default_global_redis_client()`)
        :param expiration: Cache TTL. Supports:
            - `int` or `float`: Seconds.
            - `timedelta`: Fixed duration.
            - `tuple[timedelta, timedelta]`: A random duration between [min, max] (jitter).
        :param value_validator: Optional function `f(blackboard[, tracer]) -> str`.
            The cache is only considered a hit if this value matches the one stored during the cache set.
            Useful for invalidating cache when dependent logic or state changes.
        :param enabled: A boolean or factory function `f(blackboard) -> bool` to toggle caching.
        :param name: Optional node name.

        Example::

            Tree()
            .Sequence()
            ._().RedisCacher(redis_client,
                             key_func=lambda b: f"user_data:{b.user_id}",
                             expiration=(timedelta(minutes=5), timedelta(minutes=10)), # Random jitter
                             value_validator=lambda b: b.version_tag # Invalidate if version changes
                            )
            ._()._().Function(fetch_expensive_data)
            .End()
        """
        return self._attach(RedisCacherNode(key_func, redis_client, expiration, value_validator, enabled, name))

    def Terminable(
        self,
        key_func: TerminableRedisKeyFunction,
        redis_client: async_redis.Redis | None = None,
        monitor_interval_ms: float = 500,  # ms
        name: str = "",
    ) -> Self:
        """
        Decorator that allows external interruption of a task via a Redis key.

        - First child: The main task to execute.
        - Second child (optional): The fallback task to execute if interrupted.

        Behavior:
        1. Simultaneously runs the main task and polls a Redis key (generated by `key_func`).
        2. If the main task finishes first, its result is returned.
        3. If the Redis key is created (external signal), the main task is cancelled immediately.
        4. Upon interruption, it executes the fallback child (if provided) or returns `FAIL(None)`.
        5. Note: The monitored Redis key is automatically deleted when the node starts and when a signal is detected.

        :param key_func: Factory function `f(blackboard) -> str` to generate the termination signal key.
        :param redis_client: An asynchronous Redis client instance.
            defaults to `_DEFAULT_GLOBAL_REDIS_INSTANCE` (which can be set by `set_default_global_redis_client()`)
        :param monitor_interval_ms: Polling interval in milliseconds to check for the Redis key.
        :param name: Optional node name.

        Example::

            # 1. Building the tree
            tree = (
                Tree()
                .Terminable(redis, lambda b: f"stop:{b.job_id}")
                ._().Function(long_running_job)      # First child
                ._().Fallback()
                ._()._().Function(on_interrupted)    # Optional fallback on interruption.
                .End()
            )

            # 2. To trigger termination from an external script or process:
            await redis.set(f"stop:{job_id}", "1")
        """
        return self._attach(TerminableDecoratorNode(key_func, redis_client, monitor_interval_ms, name))

    ##########################
    # Builder :: CompositeNode
    ##########################

    def Sequence(
        self,
        children: list[Node[B]] | None = None,
        name: str = "",
    ) -> Self:
        """
        Executes children sequentially until a failure occurs.

        - Status: Returns `OK` if all children succeed; returns `FAIL` if any child fails.
        - Data: Returns the `data` from the last successfully executed child (or `None` if the first child fails).

        Example::

            Tree()
            .Sequence()
            ._().Function(A)
            ._().Function(B)
            ._().Function(C)
            .End()

          * If A, B succeed and C fails: returns `FAIL(B.data)`.
          * If all (A, B, C) succeed: returns `OK(C.data)`.
          * If A fails: returns `FAIL(None)`.
        """
        return self._attach(SequenceNode[B](children, name))

    def Parallel(
        self,
        children: list[Node[B]] | None = None,
        concurrency_limit: int = 3,
        name: str = "",
    ) -> Self:
        """
        Executes all children concurrently.

        - Status: Returns `OK` if all children succeed; otherwise returns `FAIL`.
        - Data: The result `.data` is always `None`. Use the blackboard to collect outputs.

        :param concurrency_limit: Limits the number of concurrent tasks.

        Example::

            Tree()
            .Parallel(concurrency_limit=2)
            ._().Function(A)
            ._().Function(B)
            ._().Function(C)
            .End()

        In this example:
            * A, B, and C run concurrently.
            * Returns `OK(None)` if all succeed, or `FAIL(None)` if any child fails.
        """
        return self._attach(ParallelNode[B](children, concurrency_limit, name))

    def Gather[B1](
        self, params_factory: GatherParamsFactory[B, B1], concurrency_limit: int = 3, name: str = ""
    ) -> Self:
        """
        Concurrently executes a batch of subtrees, each with its own blackboard.

        - Status: Returns `OK` if all subtrees succeed; otherwise returns `FAIL`.
        - Data: Returns a list containing the `.data` from each subtree, following the input order.
        - Note: The count and order of provided blackboards must match the trees.

        :param params_factory: A function `f(current_blackboard) -> (list[Tree], list[Blackboard])`.
        :param concurrency_limit: Maximum number of concurrent tasks.

        Example::

            def gather_params(b: Blackboard):
                # Returns a tuple of (list_of_trees, list_of_blackboards)
                return [fetch_tree] * len(urls), [SubBlackboard(url=x) for x in urls]

            def collect_results(b: Blackboard, results: list):
                b.final_results = results

            tree = (
                Tree()
                .Sequence()
                ._().Gather(params_factory=gather_params, concurrency_limit=3)
                ._().WriteBlackboard(collect_results)
                .End()
            )
        """
        return self._attach(GatherNode[B, B1](params_factory, concurrency_limit, name))

    def Selector(
        self,
        children: list[Node[B]] | None = None,
        name: str = "",
    ) -> Self:
        """
        Executes children sequentially until the first child succeeds.

        - Status: Returns `OK` if any child succeeds; otherwise returns `FAIL`.
        - Data: Returns the result of the first successful child. If all fail, returns `FAIL(None)`.

        Example::

            Tree()
            .Selector()
            ._().Function(A)
            ._().Function(B)
            ._().Function(C)
            .End()

        In this example:
            * If B is the first to succeed, the Selector immediately returns B's result (OK).
            * If all nodes (A, B, C) fail, it returns `FAIL(None)`.
        """
        return self._attach(SelectorNode[B](children, name))

    def RandomSelector(
        self,
        children: list[Node[B]] | None = None,
        weights: list[float] | RandomWeightsFactory[B] | None = None,
        name: str = "",
    ) -> Self:
        """
        Executes children in a randomized order until the first child succeeds.

        - Status: Returns `OK` if any child succeeds; otherwise returns `FAIL`.
        - Data: Returns the result of the first successful child. If all fail, returns `FAIL(None)`.

        :param weights: None, a list of floats or a factory function `f(blackboard) -> list[float]`
            defining the selection probabilities.
            Defaults to `None`, which means to select children branches with equal probability.

        Example::

            Tree()
            .RandomSelector(weights=[0.1, 0.7, 0.2])
            ._().Function(A)
            ._().Function(B)
            ._().Function(C)
            .End()
        """
        return self._attach(RandomSelectorNode[B](children, weights, name))

    ##########################
    # Builder :: IF/Else
    ##########################

    def If(self, attr_or_condition_func: str | ConditionFunction, name: str = "") -> Self:
        """
        Conditional execution (If/Else) supporting one or two children.

        Supported condition types:
        - String: Checks the boolean value of `blackboard.<attr>`.
        - Function: Sync or async with signatures: `()`, `(blackboard)`, or `(blackboard, tracer) -> bool`.

        Behaviors:
        1. If (Single child): Returns the child's result if the condition is true; otherwise returns `OK(None)`.
        2. If/Else: Returns the first child's result if true, or the second child's result if false.

        Example (If only)::

            Tree()
            .If(lambda b: b.should_run)
            ._().Function(A)
            .End()

        Example (If/Else)::

            Tree()
            .If(lambda b: b.should_run)
            ._().Function(A)
            ._().Else()
            ._()._().Function(B)
            .End()
        """
        return self._attach(IfNode[B](attr_or_condition_func, name))

    def Else(self, name: str = "") -> Self:
        """
        Used in conjunction with `If`.
        Checks out docsting of `If` for more detail.
        """
        return self._attach(ElseNode[B](name))


#############
# Redis
##############

_DEFAULT_GLOBAL_REDIS_INSTANCE: async_redis.Redis | None


def set_default_global_redis_client(url: str, **kwargs) -> None:
    """Sets the default global redis_client.
    We ensures it's threading safe.

    Example::
        tinytasktree.set_default_global_redis_client("redis://127.0.0.1:6379")
    """
    global _DEFAULT_GLOBAL_REDIS_INSTANCE
    _DEFAULT_GLOBAL_REDIS_INSTANCE = cast(
        async_redis.Redis, ThreadLocalProxy(lambda: async_redis.Redis.from_url(url, **kwargs))
    )


#############
# Global Hooks
#############

type SpawnedTaskFinishHook = Callable[[Context, Tracer, Result], None | Awaitable[None]]

_GLOBAL_HOOK_AFTER_SPAWNED_TASK_FINISH: list[SpawnedTaskFinishHook] = []


def register_global_hook_after_spawned_task_finish(
    hook: SpawnedTaskFinishHook,
) -> None:
    """Registers a global hook called after any spawned task finishes.

    Useful for cleanup (e.g., closing db sessions) in async/parallel tasks.
    """
    _GLOBAL_HOOK_AFTER_SPAWNED_TASK_FINISH.append(hook)


async def _call_spawned_task_finish_hook(context: Context, tracer: Tracer, result: Result) -> None:
    if not _GLOBAL_HOOK_AFTER_SPAWNED_TASK_FINISH:
        return
    for hook in _GLOBAL_HOOK_AFTER_SPAWNED_TASK_FINISH:
        try:
            hook_result = hook(context, tracer, result)
            if inspect.isawaitable(hook_result):
                await hook_result
        except Exception as hook_exc:
            tracer.error(hook_exc)


#############
# Helpers
##############


def _inspect_func_parameters_count(func: Callable) -> int:
    """Inspects function's number of parameters"""
    if isinstance(func, functools.partial):
        # Avoid the special case: functools.partial(func, arg=x)
        # In which case the number of parameters should be decremented by 1.
        sig = inspect.signature(func.func)
        try:
            bound = sig.bind_partial(*func.args, **(func.keywords or {}))
        except TypeError:
            return len(sig.parameters)
        return max(0, len(sig.parameters) - len(bound.arguments))
    # The normal case
    sig = inspect.signature(func)
    return len(sig.parameters)


def _orjson_default_serializer(obj: Any):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, datetime):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(obj, date):
        return obj.strftime("%Y-%m-%d")
    elif isinstance(obj, timedelta):
        return obj.total_seconds()
    elif isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "json") and callable(obj.json):
        return obj.json()
    if hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()
    raise TypeError(f"{obj.__class__.__name__}: not json serializable")


def _try_to_string(data: Any) -> str:
    if isinstance(data, dict):
        try:
            return orjson.dumps(data).decode()
        except Exception:
            return str(data)
    if isinstance(data, list):
        return "[" + (",".join([_try_to_string(x) for x in data])) + "]"
    if is_dataclass(data):
        try:
            return orjson.dumps(data, default=_orjson_default_serializer).decode()  # type: ignore
        except Exception:
            return str(data)
    return str(data)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        num = int(value)
    except Exception:
        return None
    return num


def _merge_token_fields(target: dict[str, int], prompt: int | None, completion: int | None, total: int | None) -> None:
    if prompt is not None:
        target["prompt"] = prompt
    if completion is not None:
        target["completion"] = completion
    if total is not None:
        target["total"] = total


def _add_token_totals(total: dict[str, int], tokens: dict[str, int] | None) -> None:
    if not tokens:
        return
    for key, value in tokens.items():
        total[key] = total.get(key, 0) + int(value)


def _normalized_func_name(func: Callable) -> str:
    if isinstance(func, functools.partial):
        name = getattr(func.func, "__name__", func.func.__class__.__name__)
        name = f"{name}_partial"
    else:
        name = getattr(func, "__name__", func.__class__.__name__)
    if name == "<lambda>":
        return "lambda"
    return name


def _format_exception(e: BaseException) -> str:
    return f"{e.__class__.__name__}: {str(e)}"


def _weighted_shuffle[T](items: list[T], weights: list[float] | None = None) -> list[T]:
    if weights is None:
        return random.sample(items, len(items))
    keys = [random.random() ** (1.0 / w) for w in weights]
    return [x for _, x in sorted(zip(keys, items), reverse=True)]


class ThreadLocalProxy[T]:
    """Thread-local proxy for non-thread-safe objects.

    All attribute access is forwarded to a per-thread instance created lazily
    via the provided factory. This lets you treat a non-thread-safe object as a
    "pseudo-global" safely, because each thread gets its own instance.

    Example:
        def my_object_factory() -> MyObject:
            return MyObject(config="...")

        my_object: MyObject = cast(MyObject, ThreadLocalProxy(my_object_factory))
        my_object.do_something()
    """

    def __init__(self, factory: Callable[[], T]) -> None:
        """Create a proxy that lazily constructs one instance per thread."""
        self._local = threading.local()
        self._factory = factory

    def _get_instance(self) -> T:
        """Return this thread's instance, creating it on first access."""
        instance = getattr(self._local, "instance", None)
        if instance is None:
            instance = self._factory()
            setattr(self._local, "instance", instance)
        return instance

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the underlying thread-local instance.
        Note: this simple proxy does not forward Python "dunder" methods
        (e.g., __len__, __iter__). For many client objects, this is sufficient.
        """
        return getattr(self._get_instance(), name)


################
# Logging
################


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record):
        msg = super().format(record)
        color = self.COLORS.get(record.levelno, "")
        return f"{color}{msg}{self.RESET}"


logger = logging.getLogger(__name__)
_DISABLE_TASKTREE_LOGGING = os.getenv("DISABLE_TASKTREE_LOGGING", "").strip().lower() not in {"", "0", "false"}
if _DISABLE_TASKTREE_LOGGING:
    logger.disabled = True
else:
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("[%(levelname)s] %(asctime)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


################
# httpserver
################


def create_http_app(trace_dir: str = ".traces") -> Any:
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
    except Exception as e:  # pragma: no cover - optional dependency
        raise TasktreeError("fastapi and pydantic are required for the http server") from e

    storage = FileTraceStorageHandler(trace_dir)
    app = FastAPI()

    class LLMRequest(BaseModel):
        model: str
        messages: list[JSON]
        stream: bool = False
        api_key: str | None = None

    @app.get("/trace/{trace_id}")
    async def get_trace(trace_id: str) -> JSON:
        try:
            return await storage.query(trace_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @dataclass
    class LLMBlackboard:
        model: str
        messages: list[JSON]
        output: str = ""
        api_key: str | None = None
        stream_on_delta: LLMStreamOnChunkCallback1["LLMBlackboard"] | None = None

    def _bb_stream_on_delta(b: "LLMBlackboard", fulltext: str, delta: str, finished: bool) -> None:
        if b.stream_on_delta is not None:
            b.stream_on_delta(b, fulltext, delta, finished)

    # fmt: off
    llm_tree = (
        Tree[LLMBlackboard]("LLMHttp")
        .Sequence()
        ._().LLM(lambda b: b.model, lambda b: b.messages, stream=True, stream_on_delta=_bb_stream_on_delta, api_key=lambda b: b.api_key)
        ._().WriteBlackboard("output")
        .End()
    )
    # fmt: on

    @app.post("/llm")
    async def llm(req: LLMRequest) -> Any:
        context = Context()
        blackboard = LLMBlackboard(model=req.model, messages=req.messages, api_key=req.api_key)

        if not req.stream:
            async with context.using_blackboard(blackboard):
                result = await llm_tree(context)
            if not result.is_ok():
                raise HTTPException(status_code=500, detail=str(result))
            return {"output": blackboard.output}

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        error_msg: list[str] = []

        def on_delta(b, fulltext, delta, done):
            if delta:
                queue.put_nowait(delta)
            if done:
                queue.put_nowait(None)

        blackboard.stream_on_delta = on_delta

        async def _stream() -> AsyncGenerator[bytes, None]:
            async with context.using_blackboard(blackboard):

                async def _run() -> None:
                    try:
                        result = await llm_tree(context)
                        if not result.is_ok():
                            raise TasktreeError(str(result))
                    except Exception as e:
                        error_msg.append(str(e))
                        queue.put_nowait(f"\n[ERROR] {e}\n")
                        queue.put_nowait(None)
                        return
                    queue.put_nowait(None)

                task = asyncio.create_task(_run())
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    yield chunk.encode("utf-8")
                await task

        return StreamingResponse(_stream(), media_type="text/plain")

    return app


def run_httpserver(host: str = "127.0.0.1", port: int = 8000, trace_dir: str = ".traces") -> None:
    try:
        import uvicorn
    except Exception as e:  # pragma: no cover - optional dependency
        raise TasktreeProgrammingError("uvicorn is required to run the http server") from e
    uvicorn.run(create_http_app(trace_dir), host=host, port=port)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="tinytasktree utilities")
    parser.add_argument("--httpserver", action="store_true", help="start the built-in FastAPI server")
    parser.add_argument("--host", default="127.0.0.1", help="http server host")
    parser.add_argument("--port", type=int, default=8000, help="http server port")
    parser.add_argument("--trace-dir", default=".traces", help="trace storage directory")
    args = parser.parse_args()

    if args.httpserver:
        run_httpserver(host=args.host, port=args.port, trace_dir=args.trace_dir)
        return

    parser.print_help()


if __name__ == "__main__":
    _main()

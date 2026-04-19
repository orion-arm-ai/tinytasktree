"""LLM node tool call tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import tinytasktree


@dataclass
class Blackboard:
    prompt: str


def make_messages(b: Blackboard) -> list[tinytasktree.JSON]:
    return [{"role": "user", "content": b.prompt}]


TOOLS = [
    tinytasktree.ToolDef(
        type="function",
        function=tinytasktree.ToolFunctionDef(
            name="get_weather",
            description="Get the weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        ),
    ),
]


def _find_first_trace_by_kind(root: tinytasktree.TraceNode, kind: str) -> tinytasktree.TraceNode:
    stack = [root]
    while stack:
        node = stack.pop()
        if node.kind == kind:
            return node
        stack.extend(node.children.values())
    raise AssertionError(f"trace node kind not found: {kind}")


# --- Basic tool call ---

async def test_llm_tool_call_basic(mock_openai):
    """Test that tool_calls from LLM are executed and fed back."""
    call_count = 0

    async def handler(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "Beijing"}',
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "_hidden_params": {"response_cost": 0.0},
            }
        else:
            return {
                "choices": [
                    {
                        "message": {"content": "The weather in Beijing is sunny."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
                "_hidden_params": {"response_cost": 0.0},
            }

    mock_openai(handler=handler)
    tool_results = []

    def tool_executor(b: Blackboard, tool_call: tinytasktree.ToolCall) -> tinytasktree.JSON:
        tool_results.append(tool_call.function.name)
        return {"location": "Beijing", "weather": "sunny", "temperature": 25}

    tree = (
        tinytasktree.Tree[Blackboard]("ToolCallBasic")
        .Sequence()
        ._().LLM(
            "mock/tool",
            make_messages,
            tools=TOOLS,
            tool_executor=tool_executor,
            max_iterations=3,
        )
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="Whats the weather in Beijing?")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "The weather in Beijing is sunny."
    assert call_count == 2
    assert len(tool_results) == 1
    assert tool_results[0] == "get_weather"


async def test_llm_no_tool_executor(mock_openai):
    """Test that when LLM returns tool_calls but no tool_executor is set, output is returned."""
    call_count = 0

    async def handler(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "Shanghai"}',
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "_hidden_params": {"response_cost": 0.0},
            }
        else:
            return {
                "choices": [
                    {"message": {"content": "Shanghai is hot."}, "finish_reason": "stop"},
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
                "_hidden_params": {"response_cost": 0.0},
            }

    mock_openai(handler=handler)
    tree = (
        tinytasktree.Tree[Blackboard]("NoToolExecutor")
        .Sequence()
        ._().LLM("mock/no-tool", make_messages, tools=TOOLS)
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="Shanghai weather")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert call_count == 1


async def test_llm_tool_call_async_executor(mock_openai):
    """Test that async tool_executor works."""
    call_count = 0

    async def handler(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"query": "hello"}',
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "_hidden_params": {"response_cost": 0.0},
            }
        else:
            return {
                "choices": [
                    {"message": {"content": "done"}, "finish_reason": "stop"},
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
                "_hidden_params": {"response_cost": 0.0},
            }

    mock_openai(handler=handler)

    async def async_executor(b: Blackboard, tool_call: tinytasktree.ToolCall) -> tinytasktree.JSON:
        return {"result": "async_work"}

    tree = (
        tinytasktree.Tree[Blackboard]("AsyncExecutor")
        .Sequence()
        ._().LLM("mock/async", make_messages, tools=TOOLS, tool_executor=async_executor)
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="do something")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "done"
    assert call_count == 2


async def test_llm_tool_call_max_iterations(mock_openai):
    """Test that max_iterations limits the tool call loop."""
    call_count = 0

    async def handler(**kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{call_count}",
                                "type": "function",
                                "function": {"name": "loop_tool", "arguments": "{}"},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_openai(handler=handler)
    tool_results = []

    def tool_executor(b: Blackboard, tool_call: tinytasktree.ToolCall) -> tinytasktree.JSON:
        tool_results.append(tool_call.function.name)
        return {"status": "ok"}

    tree = (
        tinytasktree.Tree[Blackboard]("MaxIterations")
        .Sequence()
        ._().LLM("mock/loop", make_messages, tools=TOOLS, tool_executor=tool_executor, max_iterations=3)
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="loop test")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert call_count == 3
    assert len(tool_results) == 3


async def test_llm_tool_call_factory(mock_openai):
    """Test that tools can be a factory function."""
    async def handler(**kwargs):
        return {
            "choices": [
                {"message": {"content": "factory tools work"}, "finish_reason": "stop"},
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_openai(handler=handler)

    def tools_factory(b: Blackboard) -> list[tinytasktree.ToolDef]:
        return TOOLS

    tree = (
        tinytasktree.Tree[Blackboard]("ToolFactory")
        .Sequence()
        ._().LLM("mock/factory", make_messages, tools=tools_factory)
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="factory test")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "factory tools work"


# --- Multiple tools & edge cases ---

async def test_llm_tool_call_multiple_tools_in_one_response(mock_openai):
    """Test that multiple tool_calls in a single response are all executed."""
    call_count = 0

    async def handler(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"location": "Beijing"}'},
                                },
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"location": "Shanghai"}'},
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "_hidden_params": {"response_cost": 0.0},
            }
        else:
            return {
                "choices": [
                    {"message": {"content": "Both cities are hot."}, "finish_reason": "stop"},
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
                "_hidden_params": {"response_cost": 0.0},
            }

    mock_openai(handler=handler)
    tool_results = []

    def tool_executor(b: Blackboard, tool_call: tinytasktree.ToolCall) -> tinytasktree.JSON:
        tool_results.append(tool_call.function.arguments)
        return {"weather": "hot"}

    tree = (
        tinytasktree.Tree[Blackboard]("MultiTool")
        .Sequence()
        ._().LLM("mock/multi", make_messages, tools=TOOLS, tool_executor=tool_executor)
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="Weather check")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "Both cities are hot."
    assert call_count == 2
    assert len(tool_results) == 2


async def test_llm_tool_call_direct_answer(mock_openai):
    """Test that when LLM returns no tool_calls, output is returned directly."""
    call_count = 0

    async def handler(**kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "choices": [
                {"message": {"content": "No tools needed."}, "finish_reason": "stop"},
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_openai(handler=handler)

    def tool_executor(b: Blackboard, tool_call: tinytasktree.ToolCall) -> tinytasktree.JSON:
        return {"error": "should not be called"}

    tree = (
        tinytasktree.Tree[Blackboard]("DirectAnswer")
        .Sequence()
        ._().LLM("mock/direct", make_messages, tools=TOOLS, tool_executor=tool_executor)
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="Just answer")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "No tools needed."
    assert call_count == 1


async def test_llm_tool_call_no_tools_provided(mock_openai):
    """Test that LLM works normally when tools=None."""
    call_count = 0

    async def handler(**kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "choices": [
                {"message": {"content": "Hello!"}, "finish_reason": "stop"},
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_openai(handler=handler)

    tree = (
        tinytasktree.Tree[Blackboard]("NoTools")
        .Sequence()
        ._().LLM("mock/no-tools", make_messages)
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="hi")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "Hello!"


# --- Tracer ---

async def test_llm_tool_call_tracer_records_details(mock_openai):
    """Test that tracer records tool execution details."""

    async def handler(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_x",
                                "type": "function",
                                "function": {"name": "test_tool", "arguments": '{"x": 1}'},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_openai(handler=handler)

    def tool_executor(b: Blackboard, tool_call: tinytasktree.ToolCall) -> tinytasktree.JSON:
        return {"result": "ok"}

    tree = (
        tinytasktree.Tree[Blackboard]("TracerDetail")
        .Sequence()
        ._().LLM("mock/tracer", make_messages, tools=TOOLS, tool_executor=tool_executor)
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="tracer test")
    async with context.using_blackboard(blackboard):
        await tree(context)

    trace = _find_first_trace_by_kind(context.trace_root(), "LLM")
    assert trace.attributes.get("tool_calls") is True
    assert trace.attributes.get("iteration") is not None
    assert trace.attributes.get("tools") is True


async def test_llm_tool_call_iteration_count(mock_openai):
    """Test that iteration count is correctly tracked in tracer."""
    call_count = 0

    async def handler(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call_{call_count}",
                                    "type": "function",
                                    "function": {"name": "loop", "arguments": "{}"},
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "_hidden_params": {"response_cost": 0.0},
            }
        else:
            return {
                "choices": [
                    {"message": {"content": "done"}, "finish_reason": "stop"},
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
                "_hidden_params": {"response_cost": 0.0},
            }

    mock_openai(handler=handler)

    def tool_executor(b: Blackboard, tool_call: tinytasktree.ToolCall) -> tinytasktree.JSON:
        return {"status": "ok"}

    tree = (
        tinytasktree.Tree[Blackboard]("IterCount")
        .Sequence()
        ._().LLM("mock/iter", make_messages, tools=TOOLS, tool_executor=tool_executor, max_iterations=10)
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="iter test")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "done"
    assert call_count == 3


# --- Error handling ---

async def test_llm_tool_call_error_result(mock_openai):
    """Test that tool execution error results are fed back to LLM."""
    async def handler(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_err",
                                "type": "function",
                                "function": {"name": "fail_tool", "arguments": "{}"},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_openai(handler=handler)

    def tool_executor(b: Blackboard, tool_call: tinytasktree.ToolCall) -> tinytasktree.JSON:
        return {"error": "tool failed", "code": 500}

    tree = (
        tinytasktree.Tree[Blackboard]("ErrorResult")
        .Sequence()
        ._().LLM("mock/error", make_messages, tools=TOOLS, tool_executor=tool_executor)
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="error test")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data is not None


async def test_llm_tool_call_with_content_and_tool_calls(mock_openai):
    """Test that LLM can return both content text AND tool_calls."""
    async def handler(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": "Let me check the weather for you.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"location": "Tokyo"}'},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_openai(handler=handler)

    def tool_executor(b: Blackboard, tool_call: tinytasktree.ToolCall) -> tinytasktree.JSON:
        return {"city": "Tokyo", "weather": "rainy"}

    tree = (
        tinytasktree.Tree[Blackboard]("ContentAndTools")
        .Sequence()
        ._().LLM("mock/content", make_messages, tools=TOOLS, tool_executor=tool_executor)
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="weather")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data is not None


async def test_llm_tool_call_tool_executor_crash(mock_openai):
    """Test that exceptions in tool_executor do not crash the whole node."""
    async def handler(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "crash_tool", "arguments": "{}"},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_openai(handler=handler)

    def tool_executor(b: Blackboard, tool_call: tinytasktree.ToolCall) -> tinytasktree.JSON:
        raise ValueError("tool crashed!")

    tree = (
        tinytasktree.Tree[Blackboard]("ExecutorCrash")
        .Sequence()
        ._().LLM("mock/crash", make_messages, tools=TOOLS, tool_executor=tool_executor)
        .End()
    )

    context = tinytasktree.Context()
    result = await tree(context)

    assert not result.is_ok()


# --- Streaming ---

async def test_llm_tool_call_stream_with_tool_calls(mock_openai):
    """Test that streaming mode also handles tool_calls correctly."""
    call_count = 0

    async def stream_handler(**kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            async def gen():
                yield {
                    "choices": [
                        {"delta": {"content": None}, "finish_reason": None},
                    ],
                }
                yield {
                    "choices": [
                        {
                            "delta": {"tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"loc": "NYC"}'},
                                },
                            ]},
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }

            return gen()
        else:
            async def gen():
                yield {
                    "choices": [
                        {"delta": {"content": "NYC is warm."}, "finish_reason": None},
                    ],
                }
                yield {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
                }

            return gen()

    mock_openai(handler=stream_handler)
    streamed_chunks = []

    def stream_callback(b: Blackboard, full: str, delta: str, finished: bool) -> None:
        streamed_chunks.append(delta)

    def tool_executor(b: Blackboard, tool_call: tinytasktree.ToolCall) -> tinytasktree.JSON:
        return {"city": "NYC", "temp": 22}

    tree = (
        tinytasktree.Tree[Blackboard]("StreamTools")
        .Sequence()
        ._().LLM(
            "mock/stream",
            make_messages,
            stream=True,
            tools=TOOLS,
            tool_executor=tool_executor,
            stream_on_delta=stream_callback,
        )
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="streaming")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert call_count == 2
    assert "NYC is warm." in "".join(streamed_chunks)


async def test_llm_tool_call_cost_accumulation(mock_openai):
    """Test that cost is correctly accumulated across tool call loop iterations.

    Each LLM response's cost should be recorded once. The tool call loop
    should not skip cost recording for subsequent responses.
    """
    call_count = 0

    async def handler(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "Beijing"}',
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "_hidden_params": {"response_cost": 0.5},
            }
        else:
            return {
                "choices": [
                    {"message": {"content": "done"}, "finish_reason": "stop"},
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
                "_hidden_params": {"response_cost": 0.3},
            }

    mock_openai(handler=handler)

    def tool_executor(b: Blackboard, tool_call: tinytasktree.ToolCall) -> tinytasktree.JSON:
        return {"result": "ok"}

    tree = (
        tinytasktree.Tree[Blackboard]("CostAccum")
        .Sequence()
        ._().LLM("mock/cost", make_messages, tools=TOOLS, tool_executor=tool_executor)
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="cost test")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert call_count == 2

    # Each response's cost should be recorded: 0.5 + 0.3 = 0.8
    trace = _find_first_trace_by_kind(context.trace_root(), "LLM")
    assert trace.cost == pytest.approx(0.8), (
        f"Cost should be 0.8 (0.5 + 0.3), got {trace.cost}"
    )


def test_obj_get_model_dump_safety():
    """Test that _obj_get handles model_dump() returning non-dict safely.

    Regression test for the fix: _obj_get should not crash when
    model_dump() returns None or a non-dict value.
    """
    class ModelDumpNone:
        def model_dump(self):
            return None

    class ModelDumpString:
        def model_dump(self):
            return "not a dict"

    class ModelDumpDict:
        def model_dump(self):
            return {"key": "value"}

    node = tinytasktree.LLMNode("test", lambda b: [])

    # Normal dict case
    assert node._obj_get({"key": "value"}, "key") == "value"
    assert node._obj_get({"key": "value"}, "missing", "default") == "default"

    # model_dump() returns dict
    assert node._obj_get(ModelDumpDict(), "key") == "value"
    assert node._obj_get(ModelDumpDict(), "missing", "default") == "default"

    # model_dump() returns None - should NOT crash, return default
    assert node._obj_get(ModelDumpNone(), "key", "default") == "default"

    # model_dump() returns non-dict - should NOT crash, return default
    assert node._obj_get(ModelDumpString(), "key", "default") == "default"

    # None object
    assert node._obj_get(None, "key", "default") == "default"


# --- Regression tests for fix: messages copy & streaming index merge ---

async def test_llm_static_messages_not_mutated(mock_openai):
    """Test that passing a static messages list does not get mutated by LLMNode.

    When the caller passes a list directly (not via a callable factory), the
    LLMNode must not append assistant/tool messages into that list, because
    the same list object could be reused across multiple LLMNode executions.
    """
    call_count = 0

    async def handler(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "Tokyo"}',
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "_hidden_params": {"response_cost": 0.0},
            }
        else:
            return {
                "choices": [
                    {
                        "message": {"content": "Tokyo is rainy."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
                "_hidden_params": {"response_cost": 0.0},
            }

    mock_openai(handler=handler)

    # Static list — not a factory function
    static_messages = [{"role": "user", "content": "Whats the weather in Tokyo?"}]
    original_len = len(static_messages)

    def tool_executor(b: Blackboard, tool_call: tinytasktree.ToolCall) -> tinytasktree.JSON:
        return {"location": "Tokyo", "weather": "rainy"}

    tree = (
        tinytasktree.Tree[Blackboard]("StaticMessages")
        .Sequence()
        ._().LLM(
            "mock/static",
            static_messages,  # pass the list directly, not a callable
            tools=TOOLS,
            tool_executor=tool_executor,
            max_iterations=3,
        )
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="Tokyo weather")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "Tokyo is rainy."
    # The static list must not have been mutated
    assert len(static_messages) == original_len
    assert static_messages == [{"role": "user", "content": "Whats the weather in Tokyo?"}]


async def test_llm_streaming_tool_call_indexed_deltas(mock_openai):
    """Test that streaming deltas with index (no id) are correctly merged.

    Real OpenAI-compatible streaming responses often split tool_call arguments
    across multiple chunks. Later chunks may only carry `index` and partial
    `arguments` without repeating the `id`. The LLMNode must merge these by
    index when id is absent.
    """
    call_count = 0

    async def stream_handler(**kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            async def gen():
                # Chunk 1: initial tool_call with id, name, and first args
                yield {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_abc",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"loc": "',
                                        },
                                    },
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                # Chunk 2: only index + arguments (no id!)
                yield {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "arguments": 'Berlin"}',
                                        },
                                    },
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }

            return gen()
        else:
            async def gen():
                yield {
                    "choices": [
                        {"delta": {"content": "Berlin is cloudy."}, "finish_reason": None},
                    ],
                }
                yield {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
                }

            return gen()

    mock_openai(handler=stream_handler)
    executed_calls = []

    def tool_executor(b: Blackboard, tool_call: tinytasktree.ToolCall) -> tinytasktree.JSON:
        executed_calls.append(tool_call)
        return {"location": "Berlin", "weather": "cloudy"}

    tree = (
        tinytasktree.Tree[Blackboard]("StreamIndexDelta")
        .Sequence()
        ._().LLM(
            "mock/stream-index",
            make_messages,
            stream=True,
            tools=TOOLS,
            tool_executor=tool_executor,
            max_iterations=3,
        )
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="Berlin weather")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "Berlin is cloudy."
    assert call_count == 2
    assert len(executed_calls) == 1
    # The merged arguments should combine both chunks
    assert executed_calls[0].function.name == "get_weather"
    assert executed_calls[0].function.arguments == '{"loc": "Berlin"}'
    assert executed_calls[0].id == "call_abc"


async def test_llm_tool_result_serialized_as_json(mock_openai):
    """Test that tool results are serialized as valid JSON, not Python repr.

    Before the fix, str(result) produced Python repr (single quotes, True/None)
    which is not valid JSON. After the fix, json.dumps(result) produces proper
    JSON that models/gateways can parse reliably.
    """
    call_count = 0

    async def handler(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "Paris"}',
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "_hidden_params": {"response_cost": 0.0},
            }
        else:
            # Verify the second call received valid JSON in tool content
            second_call_messages = kwargs.get("messages", [])
            # Find the tool result message
            tool_msg = next((m for m in second_call_messages if m.get("role") == "tool"), None)
            assert tool_msg is not None, "tool message not found in second LLM call"
            content = tool_msg.get("content", "")
            # Must be valid JSON (parseable by json.loads)
            import json

            parsed = json.loads(content)
            assert parsed == {"location": "Paris", "weather": "sunny", "temp": 22}
            # Must use double quotes (JSON), not single quotes (Python repr)
            assert '"' in content and "'" not in content

            return {
                "choices": [
                    {"message": {"content": "Paris is nice."}, "finish_reason": "stop"},
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
                "_hidden_params": {"response_cost": 0.0},
            }

    mock_openai(handler=handler)

    def tool_executor(b: Blackboard, tool_call: tinytasktree.ToolCall) -> tinytasktree.JSON:
        # Return a dict with values that str() would render differently from json.dumps()
        return {"location": "Paris", "weather": "sunny", "temp": 22}

    tree = (
        tinytasktree.Tree[Blackboard]("JsonToolResult")
        .Sequence()
        ._().LLM(
            "mock/json-result",
            make_messages,
            tools=TOOLS,
            tool_executor=tool_executor,
            max_iterations=3,
        )
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="Paris weather")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "Paris is nice."


async def test_toolcall_index_roundtrip():
    """Test that ToolCall.index is preserved through to_dict/from_dict."""
    tc = tinytasktree.ToolCall(
        id="call_abc",
        type="function",
        function=tinytasktree.ToolFunction(name="get_weather", arguments='{"loc": "NY"}'),
        index=0,
    )
    d = tc.to_dict()
    assert d["index"] == 0
    tc2 = tinytasktree.ToolCall.from_dict(d)
    assert tc2.index == 0
    # Missing index should be None
    d_no_idx = {k: v for k, v in d.items() if k != "index"}
    tc3 = tinytasktree.ToolCall.from_dict(d_no_idx)
    assert tc3.index is None

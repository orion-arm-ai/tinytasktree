"""LLM tool call tests for the Tool base class and agentic outer loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import tinytasktree


@dataclass
class Blackboard:
    prompt: str
    messages: list[tinytasktree.JSON] = field(default_factory=list)
    done: bool = False
    enable_weather: bool = True
    tool_names: list[str] = field(default_factory=list)
    memory: dict[str, str] = field(default_factory=dict)


class WeatherTool(tinytasktree.Tool[Blackboard]):
    NAME = "get_weather"
    DESCRIPTION = "Get mock weather for a city."
    SIGNATURES = ["get_weather(city: str) -> object"]
    EXAMPLES = ['get_weather({"city": "Paris"})']
    SCHEMA = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }

    async def execute(
        self,
        blackboard: Blackboard,
        arguments: tinytasktree.JSON,
        context: tinytasktree.Context,
        tracer: tinytasktree.Tracer,
    ) -> tinytasktree.JSON:
        blackboard.tool_names.append(self.NAME)
        return {"city": arguments["city"], "weather": "sunny", "source": "mock"}


class SaveMemoryTool(tinytasktree.Tool[Blackboard]):
    NAME = "save_memory"
    DESCRIPTION = "Save a key/value pair to blackboard memory."
    SCHEMA = {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["key", "value"],
    }

    async def execute(
        self,
        blackboard: Blackboard,
        arguments: tinytasktree.JSON,
        context: tinytasktree.Context,
        tracer: tinytasktree.Tracer,
    ) -> tinytasktree.JSON:
        blackboard.tool_names.append(self.NAME)
        blackboard.memory[str(arguments["key"])] = str(arguments["value"])
        return {"saved": True, "total": len(blackboard.memory)}


class FailingTool(tinytasktree.Tool[Blackboard]):
    NAME = "fail_tool"
    DESCRIPTION = "Always fails."
    SCHEMA = {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
        "required": ["reason"],
    }

    async def execute(
        self,
        blackboard: Blackboard,
        arguments: tinytasktree.JSON,
        context: tinytasktree.Context,
        tracer: tinytasktree.Tracer,
    ) -> tinytasktree.JSON:
        blackboard.tool_names.append(self.NAME)
        raise RuntimeError(f"failed: {arguments['reason']}")


def make_messages(b: Blackboard) -> list[tinytasktree.JSON]:
    return [{"role": "user", "content": b.prompt}, *b.messages]


def _find_first_trace_by_kind(root: tinytasktree.TraceNode, kind: str) -> tinytasktree.TraceNode:
    stack = [root]
    while stack:
        node = stack.pop()
        if node.kind == kind:
            return node
        stack.extend(node.children.values())
    raise AssertionError(f"trace node kind not found: {kind}")


def _find_all_traces_by_kind(root: tinytasktree.TraceNode, kind: str) -> list[tinytasktree.TraceNode]:
    found: list[tinytasktree.TraceNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.kind == kind:
            found.append(node)
        stack.extend(node.children.values())
    return found


async def test_llm_auto_executes_tool_calls_and_emits_messages(mock_openai):
    recorded_request = {}

    async def handler(**kwargs):
        recorded_request.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_weather",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_openai(handler=handler)
    emitted: list[tinytasktree.JSON] = []

    def on_llm_message(b: Blackboard, message: tinytasktree.JSON, tracer: tinytasktree.Tracer) -> None:
        emitted.append(message)

    tree = (
        tinytasktree.Tree[Blackboard]("AutoToolCall")
        .Sequence()
        ._()
        .LLM("mock/tool", make_messages, tools=[WeatherTool()], on_llm_message=on_llm_message)
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="Weather in Paris?")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert isinstance(result.data, tinytasktree.LLMRunRecord)
    assert blackboard.tool_names == ["get_weather"]
    assert [message["role"] for message in emitted] == ["assistant", "tool"]
    assert emitted[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    tool_content = json.loads(emitted[1]["content"])
    assert tool_content["ok"] is True
    assert tool_content["result"] == {"city": "Paris", "weather": "sunny", "source": "mock"}

    record = result.data
    assert record.final_output == ""
    assert record.finish_reason == "tool_calls"
    assert [message["role"] for message in record.emitted_messages] == ["assistant", "tool"]
    assert record.tool_calls[0].function.name == "get_weather"
    assert record.tool_results[0].ok is True
    assert record.tool_results[0].result == {"city": "Paris", "weather": "sunny", "source": "mock"}
    assert record.messages == [*record.input_messages, *record.emitted_messages]

    assert recorded_request["tools"][0]["function"]["name"] == "get_weather"
    trace = _find_first_trace_by_kind(context.trace_root(), "LLM")
    assert trace.attributes["tool_calls"] == [record.tool_calls[0].to_dict()]
    assert trace.attributes["tool_results"][0]["name"] == "get_weather"
    assert trace.attributes["tool_executions"][0]["ok"] is True
    assert [message["role"] for message in trace.attributes["emitted_messages"]] == ["assistant", "tool"]


async def test_llm_outer_while_feeds_tool_result_to_next_llm_call(mock_openai):
    call_messages: list[list[tinytasktree.JSON]] = []

    async def handler(**kwargs):
        messages = kwargs["messages"]
        call_messages.append(messages)
        if any(message.get("role") == "tool" for message in messages):
            tool_message = next(message for message in messages if message.get("role") == "tool")
            assert json.loads(tool_message["content"])["result"]["weather"] == "sunny"
            return {
                "choices": [{"message": {"content": "Paris is sunny."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
                "_hidden_params": {"response_cost": 0.0},
            }
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_weather",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_openai(handler=handler)

    def on_llm_message(b: Blackboard, message: tinytasktree.JSON, tracer: tinytasktree.Tracer) -> None:
        b.messages.append(message)

    async def decide_next_step(
        b: Blackboard,
        tracer: tinytasktree.Tracer,
        context: tinytasktree.Context,
    ) -> tinytasktree.Result:
        record = context._last_result.data
        if record.tool_calls:
            b.done = False
            return tinytasktree.Result.OK(None)
        b.done = True
        return tinytasktree.Result.OK(record.final_output)

    tree = (
        tinytasktree.Tree[Blackboard]("AgenticToolLoop")
        .While(lambda b: not b.done, max_loop_times=4)
        ._()
        .Sequence()
        ._()
        ._()
        .LLM("mock/tool-loop", make_messages, tools=[WeatherTool()], on_llm_message=on_llm_message)
        ._()
        ._()
        .Function(decide_next_step)
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="Weather in Paris?")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "Paris is sunny."
    assert len(call_messages) == 2
    assert [message["role"] for message in blackboard.messages] == ["assistant", "tool", "assistant"]


async def test_llm_executes_multiple_tool_calls_in_order(mock_openai):
    async def handler(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": "I will save and check weather.",
                        "tool_calls": [
                            {
                                "id": "call_memory",
                                "type": "function",
                                "function": {
                                    "name": "save_memory",
                                    "arguments": '{"key": "city", "value": "Paris"}',
                                },
                            },
                            {
                                "id": "call_weather",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_openai(handler=handler)
    emitted_roles_and_names: list[tuple[str, str | None]] = []

    def on_llm_message(b: Blackboard, message: tinytasktree.JSON, tracer: tinytasktree.Tracer) -> None:
        emitted_roles_and_names.append((str(message["role"]), message.get("name")))

    tree = (
        tinytasktree.Tree[Blackboard]("MultipleToolCalls")
        .Sequence()
        ._()
        .LLM(
            "mock/multi",
            make_messages,
            tools=[SaveMemoryTool(), WeatherTool()],
            on_llm_message=on_llm_message,
        )
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="Remember Paris and get its weather.")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert blackboard.tool_names == ["save_memory", "get_weather"]
    assert emitted_roles_and_names == [
        ("assistant", None),
        ("tool", "save_memory"),
        ("tool", "get_weather"),
    ]
    assert [tool_result.name for tool_result in result.data.tool_results] == ["save_memory", "get_weather"]
    assert blackboard.memory == {"city": "Paris"}


async def test_llm_unknown_tool_call_becomes_failed_tool_message(mock_openai):
    async def handler(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_missing",
                                "type": "function",
                                "function": {
                                    "name": "missing_tool",
                                    "arguments": '{"x": 1}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_openai(handler=handler)

    tree = (
        tinytasktree.Tree[Blackboard]("UnknownToolCall")
        .Sequence()
        ._()
        .LLM("mock/missing-tool", make_messages, tools=[WeatherTool()])
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="Call missing tool")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    record = result.data
    assert record.tool_results[0].ok is False
    assert record.tool_results[0].error == "unknown tool: missing_tool"
    tool_message = record.emitted_messages[1]
    tool_content = json.loads(tool_message["content"])
    assert tool_content["ok"] is False
    assert tool_content["code"] == "tool_not_found"


async def test_llm_streaming_tool_call_deltas_are_merged_and_executed(mock_openai):
    async def handler(**kwargs):
        async def gen():
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_weather",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"city":'},
                                }
                            ]
                        }
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": ' "Paris"}'},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }

        return gen()

    mock_openai(handler=handler)
    stream_events: list[tuple[str, str, bool, str]] = []

    def on_delta(b: Blackboard, full: str, delta: str, finished: bool, reason: str) -> None:
        stream_events.append((full, delta, finished, reason))

    tree = (
        tinytasktree.Tree[Blackboard]("StreamingToolCall")
        .Sequence()
        ._()
        .LLM("mock/stream-tool", make_messages, stream=True, stream_on_delta=on_delta, tools=[WeatherTool()])
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="Weather in Paris?")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    record = result.data
    assert record.tool_calls[0].function.arguments == '{"city": "Paris"}'
    assert record.tool_results[0].ok is True
    assert blackboard.tool_names == ["get_weather"]
    assert stream_events[-1] == ("", "", True, "tool_calls")


async def test_llm_dynamic_tools_factory_uses_blackboard_state(mock_openai):
    recorded_tool_names: list[list[str]] = []

    async def handler(**kwargs):
        recorded_tool_names.append([tool["function"]["name"] for tool in kwargs["tools"]])
        if kwargs["tools"]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_weather",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Paris"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
                "_hidden_params": {"response_cost": 0.0},
            }
        return {
            "choices": [{"message": {"content": "No tools enabled."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            "_hidden_params": {"response_cost": 0.0},
        }

    def make_tools(b: Blackboard) -> list[tinytasktree.Tool[Blackboard]]:
        return [WeatherTool()] if b.enable_weather else []

    mock_openai(handler=handler)
    tree = (
        tinytasktree.Tree[Blackboard]("DynamicTools")
        .Sequence()
        ._()
        .LLM("mock/dynamic-tools", make_messages, tools=make_tools)
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="Weather?", enable_weather=True)
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert blackboard.tool_names == ["get_weather"]
    assert recorded_tool_names[-1] == ["get_weather"]

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="Weather?", enable_weather=False)
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data.final_output == "No tools enabled."
    assert blackboard.tool_names == []
    assert recorded_tool_names[-1] == []


async def test_llm_async_on_llm_message_is_called_for_each_emitted_message(mock_openai):
    async def handler(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": "I will save and check weather.",
                        "tool_calls": [
                            {
                                "id": "call_memory",
                                "type": "function",
                                "function": {
                                    "name": "save_memory",
                                    "arguments": '{"key": "city", "value": "Paris"}',
                                },
                            },
                            {
                                "id": "call_weather",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_openai(handler=handler)
    emitted_roles: list[str] = []

    async def on_llm_message(b: Blackboard, message: tinytasktree.JSON, tracer: tinytasktree.Tracer) -> None:
        emitted_roles.append(str(message["role"]))
        b.messages.append(message)

    tree = (
        tinytasktree.Tree[Blackboard]("AsyncLLMMessage")
        .Sequence()
        ._()
        .LLM(
            "mock/async-message",
            make_messages,
            tools=[SaveMemoryTool(), WeatherTool()],
            on_llm_message=on_llm_message,
        )
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="Remember Paris and get weather.")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert emitted_roles == ["assistant", "tool", "tool"]
    assert [message["role"] for message in blackboard.messages] == ["assistant", "tool", "tool"]
    assert [tool_result.name for tool_result in result.data.tool_results] == ["save_memory", "get_weather"]


async def test_llm_tool_execution_exception_becomes_failed_tool_message(mock_openai):
    async def handler(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_fail",
                                "type": "function",
                                "function": {
                                    "name": "fail_tool",
                                    "arguments": '{"reason": "boom"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_openai(handler=handler)
    tree = (
        tinytasktree.Tree[Blackboard]("FailingToolCall")
        .Sequence()
        ._()
        .LLM("mock/failing-tool", make_messages, tools=[FailingTool()])
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="Call failing tool")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    record = result.data
    assert blackboard.tool_names == ["fail_tool"]
    assert record.tool_results[0].ok is False
    assert record.tool_results[0].error == "failed: boom"
    tool_content = json.loads(record.emitted_messages[1]["content"])
    assert tool_content["ok"] is False
    assert tool_content["code"] == "tool_execution_error"


async def test_llm_invalid_tool_arguments_becomes_failed_tool_message(mock_openai):
    async def handler(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_weather",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '["Paris"]',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_openai(handler=handler)
    tree = (
        tinytasktree.Tree[Blackboard]("InvalidToolArguments")
        .Sequence()
        ._()
        .LLM("mock/invalid-tool-arguments", make_messages, tools=[WeatherTool()])
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="Call weather with invalid args")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    record = result.data
    assert blackboard.tool_names == []
    assert record.tool_results[0].ok is False
    assert record.tool_results[0].arguments == {}
    assert record.tool_results[0].error == "tool arguments must be a JSON object"
    tool_content = json.loads(record.emitted_messages[1]["content"])
    assert tool_content["code"] == "tool_execution_error"


async def test_llm_streaming_multiple_tool_call_deltas_are_merged_by_index(mock_openai):
    async def handler(**kwargs):
        async def gen():
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_memory",
                                    "type": "function",
                                    "function": {"name": "save_memory", "arguments": '{"key": '},
                                },
                                {
                                    "index": 1,
                                    "id": "call_weather",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"city": '},
                                },
                            ]
                        }
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '"city", "value": "Paris"}'},
                                },
                                {
                                    "index": 1,
                                    "function": {"arguments": '"Paris"}'},
                                },
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
            }

        return gen()

    mock_openai(handler=handler)
    tree = (
        tinytasktree.Tree[Blackboard]("StreamingMultipleToolCalls")
        .Sequence()
        ._()
        .LLM(
            "mock/stream-multi-tool",
            make_messages,
            stream=True,
            tools=[SaveMemoryTool(), WeatherTool()],
        )
        .End()
    )

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="Remember Paris and get weather.")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    record = result.data
    assert [tool_call.function.name for tool_call in record.tool_calls] == ["save_memory", "get_weather"]
    assert [tool_call.function.arguments for tool_call in record.tool_calls] == [
        '{"key": "city", "value": "Paris"}',
        '{"city": "Paris"}',
    ]
    assert [tool_result.name for tool_result in record.tool_results] == ["save_memory", "get_weather"]
    assert blackboard.memory == {"city": "Paris"}
    assert blackboard.tool_names == ["save_memory", "get_weather"]

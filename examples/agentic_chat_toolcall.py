"""Agentic chat example with persisted context messages and mock tools.

Each user turn runs the tree once. The tree loops over one-call LLM nodes and
tool execution steps, then stores the full transcript for the next turn.

Defaults:
    CHAT_STREAM=1
    TRACE_DIR=.traces
"""

import asyncio
import ast
import json
import operator
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")

from tinytasktree import Context, FileTraceStorageHandler, JSON, LLMModel, LLMProvider, LLMRunRecord, Result, Tool, Tracer, Tree


LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
CHAT_STREAM = os.getenv("CHAT_STREAM", "1").strip().lower() not in {"0", "false", "no", "off"}
TRACE_DIR = os.getenv("TRACE_DIR", ".traces").strip()
PROVIDER = LLMProvider(base_url=LLM_BASE_URL or "", api_key=LLM_API_KEY)
MODEL = LLMModel("deepseek-v4-flash", provider=PROVIDER, extra_body={"reasoning": {"enabled": False}})


@dataclass
class Blackboard:
    user_input: str = ""
    # Persisted conversation context: user, assistant, assistant tool_calls, and tool results.
    messages: list[JSON] = field(default_factory=list)
    memory: dict[str, str] = field(default_factory=dict)
    include_user_input: bool = True
    turn_finished: bool = True
    last_record: LLMRunRecord | None = None
    streaming_answer_started: bool = False


class MockWeatherTool(Tool[Blackboard]):
    NAME = "get_weather"
    DESCRIPTION = "Get mock weather for a city."
    SIGNATURES = ["get_weather(city: str) -> object"]
    EXAMPLES = ['get_weather({"city": "Tokyo"})']
    SCHEMA = {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name."}},
        "required": ["city"],
    }

    async def execute(self, blackboard: Blackboard, arguments: JSON, context: Context, tracer: Tracer) -> JSON:
        city = str(arguments["city"]).strip()
        presets = {
            "beijing": {"condition": "clear", "temperature_c": 24},
            "shanghai": {"condition": "humid", "temperature_c": 27},
            "tokyo": {"condition": "cloudy", "temperature_c": 22},
            "san francisco": {"condition": "foggy", "temperature_c": 16},
        }
        weather = presets.get(city.lower(), {"condition": "sunny", "temperature_c": 25})
        return {"city": city, **weather, "source": "mock"}


class CalculatorTool(Tool[Blackboard]):
    NAME = "calculate"
    DESCRIPTION = "Evaluate a simple arithmetic expression."
    SIGNATURES = ["calculate(expression: str) -> object"]
    EXAMPLES = ['calculate({"expression": "42 * 17 + 8"})']
    SCHEMA = {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "Arithmetic expression."}},
        "required": ["expression"],
    }

    _OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    async def execute(self, blackboard: Blackboard, arguments: JSON, context: Context, tracer: Tracer) -> JSON:
        expression = str(arguments["expression"])
        value = self._eval(ast.parse(expression, mode="eval").body)
        return {"expression": expression, "result": value}

    def _eval(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in self._OPS:
            return self._OPS[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._OPS:
            return self._OPS[type(node.op)](self._eval(node.operand))
        raise ValueError("unsupported expression")


class SaveMemoryTool(Tool[Blackboard]):
    NAME = "save_memory"
    DESCRIPTION = "Save a stable user preference or fact into in-memory key-value memory."
    SIGNATURES = ["save_memory(key: str, value: str) -> object"]
    EXAMPLES = ['save_memory({"key": "preferred_style", "value": "concise"})']
    SCHEMA = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Memory key."},
            "value": {"type": "string", "description": "Memory value."},
        },
        "required": ["key", "value"],
    }

    async def execute(self, blackboard: Blackboard, arguments: JSON, context: Context, tracer: Tracer) -> JSON:
        key = str(arguments["key"]).strip()
        value = str(arguments["value"])
        blackboard.memory[key] = value
        return {"saved": True, "key": key, "value": value, "total_memory_items": len(blackboard.memory)}


class ListMemoryTool(Tool[Blackboard]):
    NAME = "list_memory"
    DESCRIPTION = "List saved in-memory user preferences and facts."
    SIGNATURES = ["list_memory() -> object"]
    EXAMPLES = ["list_memory({})"]
    SCHEMA = {"type": "object", "properties": {}, "required": []}

    async def execute(self, blackboard: Blackboard, arguments: JSON, context: Context, tracer: Tracer) -> JSON:
        return {"memory": dict(sorted(blackboard.memory.items())), "total": len(blackboard.memory)}


def make_tools(b: Blackboard) -> list[Tool[Blackboard]]:
    tools: list[Tool[Blackboard]] = [MockWeatherTool(), CalculatorTool()]
    if b.memory:
        tools.append(ListMemoryTool())
    tools.append(SaveMemoryTool())
    return tools


def make_messages(b: Blackboard) -> list[JSON]:
    system = {
        "role": "system",
        "content": (
            "You are an agentic chat assistant. Chat history is already in context. "
            "Use tools for external actions only: weather, calculation, or saving/listing stable user facts. "
            "If a user asks weather or math, call the relevant tool before answering. "
            "If the user states a durable preference or fact, call save_memory before answering."
        ),
    }
    messages = [system, *b.messages]
    if b.include_user_input:
        messages.append({"role": "user", "content": b.user_input})
    return messages


def store_llm_record(b: Blackboard, record: LLMRunRecord, tracer: Tracer) -> None:
    b.last_record = record


def on_stream_delta(b: Blackboard, full: str, delta: str, finished: bool, reason: str = "") -> None:
    if reason == "tool_calls":
        return
    if delta:
        if not b.streaming_answer_started:
            print("Assistant: ", end="", flush=True)
            b.streaming_answer_started = True
        print(delta, end="", flush=True)
    if finished and b.streaming_answer_started:
        print()


async def process_llm_record(b: Blackboard, tracer: Tracer, context: Context) -> Result:
    record = b.last_record
    if record is None:
        b.turn_finished = True
        return Result.FAIL("missing llm record")

    b.messages = [message for message in record.messages if message.get("role") != "system"]
    b.include_user_input = False

    if not record.tool_calls:
        b.turn_finished = True
        tracer.update_attributes(chat_transcript=b.messages)
        context.parent_tracer(2).update_attributes(chat_transcript=b.messages)
        return Result.OK(record.final_output)

    tools = {tool.NAME: tool for tool in make_tools(b)}
    tool_executions: list[JSON] = []
    for tool_call in record.tool_calls:
        name = tool_call.function.name
        arguments_text = tool_call.function.arguments or "{}"
        executed = f"{name}({arguments_text})"
        started = asyncio.get_running_loop().time()
        arguments: JSON = {}
        try:
            parsed_arguments = json.loads(arguments_text)
            if not isinstance(parsed_arguments, dict):
                raise ValueError("tool arguments must be a JSON object")
            arguments = parsed_arguments
            tool = tools[name]
            executed = tool.format_call(arguments)
            result = await tool.execute(b, arguments, Context(), tracer)
            content: JSON = {"ok": True, "executed": executed, "result": result}
        except KeyError:
            content = {"ok": False, "executed": executed, "error": f"unknown tool: {name}", "code": "tool_not_found"}
        except Exception as e:
            content = {"ok": False, "executed": executed, "error": str(e), "code": "tool_execution_error"}
        duration_ms = (asyncio.get_running_loop().time() - started) * 1000
        content["duration_ms"] = duration_ms
        b.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(content, ensure_ascii=False),
            }
        )
        tool_executions.append(
            {
                "id": tool_call.id,
                "name": name,
                "arguments": arguments,
                "result": content.get("result", content),
                "ok": content["ok"],
                "error": content.get("error"),
                "iteration": record.iterations,
                "duration_ms": duration_ms,
            }
        )
    tracer.update_attributes(tool_executions=tool_executions)
    tracer.update_attributes(chat_transcript=b.messages)
    context.parent_tracer(2).update_attributes(chat_transcript=b.messages)
    return Result.OK(None)


tree = (
    Tree[Blackboard]("AgenticChat")
    .While(lambda b: not b.turn_finished, max_loop_times=8)
    ._().Sequence()
    ._()._().LLM(
        MODEL,
        make_messages,
        stream=CHAT_STREAM,
        stream_on_delta=on_stream_delta if CHAT_STREAM else None,
        tools=make_tools,
        on_llm_record=store_llm_record,
    )
    ._()._().Function(process_llm_record)
    .End()
)


async def run_turn(blackboard: Blackboard, text: str, trace_dir: str | None = None) -> str:
    blackboard.user_input = text
    blackboard.include_user_input = True
    blackboard.turn_finished = False
    blackboard.last_record = None
    blackboard.streaming_answer_started = False
    context = Context()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    if trace_dir is not None and trace_dir != "":
        storage = FileTraceStorageHandler(trace_dir)
        trace_id = await storage.save(context.trace_root())
        print(f"Trace URL: http://127.0.0.1:8000/{trace_id}")

    return str(result.data or "")


async def main() -> None:
    blackboard = Blackboard()
    mode = "streaming" if CHAT_STREAM else "non-streaming"
    trace_text = TRACE_DIR if TRACE_DIR else "disabled"
    print(f"Agentic chat ({mode}, trace_dir={trace_text}). Type 'exit' to quit.")
    while True:
        text = input("\nYou: ").strip()
        if text.lower() in {"exit", "quit"}:
            return
        if not text:
            continue
        reply = await run_turn(blackboard, text, trace_dir=TRACE_DIR)
        if not CHAT_STREAM or not blackboard.streaming_answer_started:
            print(f"Assistant: {reply}")


if __name__ == "__main__":
    asyncio.run(main())

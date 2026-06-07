"""Tool Call Example - Task Assistant

Demonstrates LLM tool call support with multiple tools:
- `get_current_time`: Returns the current date/time
- `calculate`: Performs arithmetic calculations
- `add_todo`: Adds a task to a todo list stored on the blackboard
- `list_todos`: Lists all tasks on the todo list

Each LLM node call executes requested tools once. An outer While node calls
the LLM again with tool results until it produces a final text response.

Usage:
    export LLM_BASE_URL="https://your-api.example.com/v1"
    export LLM_API_KEY="your-key"
    python example/tool_call_assistant.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")

from dataclasses import dataclass, field

from tinytasktree import (
    Context,
    FileTraceStorageHandler,
    JSON,
    LLMModel,
    LLMProvider,
    LLMRunRecord,
    Result,
    Tool,
    Tracer,
    Tree,
)

# --- Configuration ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
PROVIDER = LLMProvider(base_url=LLM_BASE_URL or "", api_key=LLM_API_KEY)
MODEL = LLMModel("deepseek-v4-flash", provider=PROVIDER, extra_body={"reasoning": {"enabled": False}})


# --- Blackboard (shared state) ---

@dataclass
class TodoItem:
    id: int
    text: str
    done: bool = False


@dataclass
class Blackboard:
    """Shared state accessible by tools and the LLM."""

    prompt: str
    messages: list[JSON] = field(default_factory=list)
    todos: list[TodoItem] = field(default_factory=list)
    done: bool = False


# --- Tools ---

class CurrentTimeTool(Tool[Blackboard]):
    NAME = "get_current_time"
    DESCRIPTION = "Get the current date and time in ISO format"
    SIGNATURES = ["get_current_time() -> object"]
    EXAMPLES = ["get_current_time({})"]
    SCHEMA = {"type": "object", "properties": {}, "required": []}

    async def execute(self, blackboard: Blackboard, arguments: JSON, context: Context, tracer) -> JSON:
        from datetime import datetime

        print(f"  [tool] {self.format_call(arguments)}")
        return {"now": datetime.utcnow().isoformat()}


class CalculatorTool(Tool[Blackboard]):
    NAME = "calculate"
    DESCRIPTION = "Perform arithmetic. Supports +, -, *, /, //, %, **"
    SIGNATURES = ["calculate(expression: str) -> object"]
    EXAMPLES = ['calculate({"expression": "2 + 3 * 4"})']
    SCHEMA = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A math expression, e.g. '2 + 3 * 4'",
            },
        },
        "required": ["expression"],
    }

    async def execute(self, blackboard: Blackboard, arguments: JSON, context: Context, tracer) -> JSON:
        print(f"  [tool] {self.format_call(arguments)}")
        expr = str(arguments["expression"])
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
        }
        try:
            result = eval(expr, {"__builtins__": {}}, allowed_names)
            return {"expression": expr, "result": result}
        except Exception as e:
            return {"error": f"calculation failed: {e}"}


class AddTodoTool(Tool[Blackboard]):
    NAME = "add_todo"
    DESCRIPTION = "Add a new task to the todo list"
    SIGNATURES = ["add_todo(text: str) -> object"]
    EXAMPLES = ['add_todo({"text": "Review PR"})']
    SCHEMA = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The task description",
            },
        },
        "required": ["text"],
    }

    async def execute(self, blackboard: Blackboard, arguments: JSON, context: Context, tracer) -> JSON:
        print(f"  [tool] {self.format_call(arguments)}")
        item = TodoItem(id=len(blackboard.todos) + 1, text=str(arguments["text"]))
        blackboard.todos.append(item)
        return {"added": item.id, "text": item.text, "total": len(blackboard.todos)}


class ListTodosTool(Tool[Blackboard]):
    NAME = "list_todos"
    DESCRIPTION = "List all tasks in the todo list"
    SIGNATURES = ["list_todos() -> object"]
    EXAMPLES = ["list_todos({})"]
    SCHEMA = {"type": "object", "properties": {}, "required": []}

    async def execute(self, blackboard: Blackboard, arguments: JSON, context: Context, tracer) -> JSON:
        print(f"  [tool] {self.format_call(arguments)}")
        if not blackboard.todos:
            return {"todos": [], "message": "No tasks yet."}
        items = [{"id": t.id, "text": t.text, "done": t.done} for t in blackboard.todos]
        return {"todos": items, "total": len(items)}


TOOLS = [CurrentTimeTool(), CalculatorTool(), AddTodoTool(), ListTodosTool()]


# --- Message Builder ---

def make_messages(b: Blackboard) -> list[JSON]:
    return [
        {"role": "system", "content": "Use tools when needed, then answer after seeing tool results."},
        *b.messages,
    ]


def on_llm_message(b: Blackboard, message: JSON, tracer: Tracer) -> None:
    b.messages.append(message)


async def decide_next_step(b: Blackboard, tracer: Tracer, context: Context) -> Result:
    result = context._last_result
    if result is None or not isinstance(result.data, LLMRunRecord):
        b.done = True
        return Result.FAIL("missing llm record")

    if result.data.tool_calls:
        b.done = False
        return Result.OK(None)

    b.done = True
    return Result.OK(result.data.final_output)


# --- Build the Tree ---

# fmt: off
tree = (
    Tree[Blackboard]("ToolCallAssistant")
    .While(lambda b: not b.done, max_loop_times=6)
    ._().Sequence()
    ._()._().LLM(
        MODEL,
        make_messages,
        tools=TOOLS,
        on_llm_message=on_llm_message,
    )
    ._()._().Function(decide_next_step)
    .End()
)
# fmt: on


# --- Main ---

async def main() -> None:
    # Example prompt that requires multiple tool calls
    prompt = """I need help with a few things:
1. What's the current time?
2. Calculate 42 * 17 + 8
3. Add 'Review PR #123' to my todo list
4. Add 'Write docs' to my todo list
5. Show me all my todos
"""

    blackboard = Blackboard(prompt=prompt)
    blackboard.messages.append({"role": "user", "content": prompt})
    context = Context()

    print("=== Tool Call Assistant ===\n")
    print(f"Prompt:\n{prompt}\n")
    print("--- Tool Execution Log ---")

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("\n--- Final Result ---")
    print(f"Response: {result.data or ''}")
    print(f"Todos: {blackboard.todos}")

    # Save trace for visualization
    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print(f"\nTrace URL: http://127.0.0.1:8000/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())

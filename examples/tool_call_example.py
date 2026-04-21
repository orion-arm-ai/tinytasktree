"""Tool call example.

Demonstrates LLM tool call support: the LLM can request tool execution,
and the tool_executor handles the actual work. The LLM is called again
with the tool results until it returns a final text response.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")

from dataclasses import dataclass

from tinytasktree import JSON, Context, FileTraceStorageHandler, LLMModel, LLMProvider, ToolCall, ToolDef, ToolFunctionDef, Tree

# Requirements:
#   - LLM_BASE_URL and LLM_API_KEY set for your LLM service
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
PROVIDER = LLMProvider(base_url=LLM_BASE_URL or "", api_key=LLM_API_KEY)
MODEL = LLMModel("qwen/qwen3.6-plus", provider=PROVIDER, extra_body={"reasoning": {"enabled": False}})


@dataclass
class Blackboard:
    prompt: str
    weather: JSON | None = None


# Define tools using dataclasses
TOOLS = [
    ToolDef(
        type="function",
        function=ToolFunctionDef(
            name="get_weather",
            description="Get the current weather in a given location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                },
                "required": ["location"],
            },
        ),
    ),
]


def make_messages(b: Blackboard) -> list[JSON]:
    return [{"role": "user", "content": b.prompt}]


def tool_executor(b: Blackboard, tool_call: ToolCall) -> JSON:
    """Execute a tool call and return the result."""
    tool_name = tool_call.function.name
    tool_args = tool_call.function.arguments
    print(f"Executing tool: {tool_name}({tool_args})")

    # Simulate weather API call
    if tool_name == "get_weather":
        return {"location": "San Francisco", "weather": "sunny", "temperature": 25}
    return {"error": f"unknown tool: {tool_name}"}


# fmt: off
tree = (
    Tree[Blackboard]("ToolCall")
    .Sequence()
    ._().LLM(
        MODEL,
        make_messages,
        tools=TOOLS,
        tool_executor=tool_executor,
        max_iterations=5,
    )
    ._().WriteBlackboard("weather")
    .End()
)
# fmt: on


async def main() -> None:
    blackboard = Blackboard(prompt="What's the weather in San Francisco?")
    context = Context()

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("Result:", result)
    print("Weather:", blackboard.weather)

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:8000/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())

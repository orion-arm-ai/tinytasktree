"""Tool call example.

Demonstrates LLM tool call support: the LLM can request tool execution,
and Tool subclasses handle the actual work. The LLM is called again
with the tool results until it returns a final text response.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")

from dataclasses import dataclass

from tinytasktree import JSON, Context, FileTraceStorageHandler, LLMModel, LLMProvider, Tool, Tree

# Requirements:
#   - LLM_BASE_URL and LLM_API_KEY set for your LLM service
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
PROVIDER = LLMProvider(base_url=LLM_BASE_URL or "", api_key=LLM_API_KEY)
MODEL = LLMModel("deepseek-v4-flash", provider=PROVIDER, extra_body={"reasoning": {"enabled": False}})


@dataclass
class Blackboard:
    prompt: str
    llm_record: object | None = None


class WeatherTool(Tool[Blackboard]):
    NAME = "get_weather"
    DESCRIPTION = "Get the current weather in a given location"
    SIGNATURES = ["get_weather(location: str) -> object"]
    EXAMPLES = ['get_weather({"location": "San Francisco, CA"})']
    SCHEMA = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
        },
        "required": ["location"],
    }

    async def execute(self, blackboard: Blackboard, arguments: JSON, context: Context, tracer) -> JSON:
        location = str(arguments.get("location") or "San Francisco")
        print(f"Executing tool: {self.format_call(arguments)}")
        return {"location": location, "weather": "sunny", "temperature": 25}


def make_messages(b: Blackboard) -> list[JSON]:
    return [{"role": "user", "content": b.prompt}]


# fmt: off
tree = (
    Tree[Blackboard]("ToolCall")
    .Sequence()
    ._().LLM(
        MODEL,
        make_messages,
        tools=[WeatherTool()],
    )
    ._().WriteBlackboard("llm_record")
    .End()
)
# fmt: on


async def main() -> None:
    blackboard = Blackboard(prompt="What's the weather in San Francisco?")
    context = Context()

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("Result:", result)
    print("Record:", blackboard.llm_record)
    if result.data:
        print("Final output:", result.data.final_output)

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:8000/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())

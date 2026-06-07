"""Tool call example.

Demonstrates LLM tool call support: the LLM node executes requested tools,
and an outer While calls the LLM again with tool results until it returns
a final text response.
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

# Requirements:
#   - LLM_BASE_URL and LLM_API_KEY set for your LLM service
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
PROVIDER = LLMProvider(base_url=LLM_BASE_URL or "", api_key=LLM_API_KEY)
MODEL = LLMModel("deepseek-v4-flash", provider=PROVIDER, extra_body={"reasoning": {"enabled": False}})


@dataclass
class Blackboard:
    prompt: str
    messages: list[JSON] = field(default_factory=list)
    done: bool = False
    llm_record: LLMRunRecord | None = None


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

    b.llm_record = result.data
    if result.data.tool_calls:
        b.done = False
        return Result.OK(None)

    b.done = True
    return Result.OK(result.data.final_output)


# fmt: off
tree = (
    Tree[Blackboard]("ToolCall")
    .While(lambda b: not b.done, max_loop_times=4)
    ._().Sequence()
    ._()._().LLM(
        MODEL,
        make_messages,
        tools=[WeatherTool()],
        on_llm_message=on_llm_message,
    )
    ._()._().Function(decide_next_step)
    .End()
)
# fmt: on


async def main() -> None:
    blackboard = Blackboard(prompt="What's the weather in San Francisco?")
    blackboard.messages.append({"role": "user", "content": blackboard.prompt})
    context = Context()

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("Result:", result)
    print("Record:", blackboard.llm_record)
    print("Final output:", result.data)

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:8000/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())

"""Log node example with an LLM call.

Runs a small tree that logs a message before and after the LLM call.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensures tinytasktree is in module path

import asyncio
from dataclasses import dataclass

from tinytasktree import Context, FileTraceStorageHandler, LLMModel, LLMProvider, Tree

# Running this example requires setting `LLM_BASE_URL` and `LLM_API_KEY`.
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
PROVIDER = LLMProvider(base_url=LLM_BASE_URL or "", api_key=LLM_API_KEY)
MODEL = LLMModel("qwen/qwen3.6-plus", provider=PROVIDER, extra_body={"reasoning": {"enabled": False}})


@dataclass
class Blackboard:
    prompt: str
    response: str = ""


def make_messages(b: Blackboard) -> list[dict[str, str]]:
    return [{"role": "user", "content": b.prompt}]


def write_response(b: Blackboard, data: str) -> None:
    b.response = data


# fmt: off
tree = (
    Tree[Blackboard]("LogLLM")
    .Sequence()
    ._().Log("About to call LLM")
    ._().LLM(MODEL, make_messages)
    ._().WriteBlackboard(write_response)
    ._().Log(lambda b: f"LLM response length: {len(b.response)} chars")
    ._().Log("LLM call done")
    .End()
)
# fmt: on


async def main() -> None:
    context = Context()
    blackboard = Blackboard(prompt="Write a short haiku about trees.")

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("Result:", result)
    print("Response:", blackboard.response)

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:8000/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())

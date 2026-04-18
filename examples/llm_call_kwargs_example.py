"""LLM call with explicit OpenAI client kwargs, request kwargs, and extra_body.

Shows how to pass arbitrary keyword arguments (e.g. temperature, max_tokens)
through `Tree.LLM`, while provider-specific request extensions go through
`extra_body`.
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
MODEL = LLMModel(
    "qwen/qwen3.6-plus",
    provider=PROVIDER,
    extra_body={"reasoning": {"enabled": False}},
    llm_call_kwargs={
        "temperature": 0.2,
        "max_tokens": 128,
        "top_p": 0.9,
    },
)


@dataclass
class Blackboard:
    prompt: str
    response: str = ""


def write_response(b: Blackboard, data: str) -> None:
    b.response = data


def make_messages(b: Blackboard) -> list[dict]:
    return [{"role": "user", "content": b.prompt}]


# fmt: off
tree = (
    Tree[Blackboard]("LLMCallKwargs")
    .Sequence()
    ._().LLM(MODEL, make_messages)
    ._().WriteBlackboard(write_response)
    .End()
)
# fmt: on


async def main() -> None:
    context = Context()
    blackboard = Blackboard(prompt="Summarize the benefits of unit testing in 3 sentences.")

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("Result:", result)
    print("Response:", blackboard.response)

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:8000/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())

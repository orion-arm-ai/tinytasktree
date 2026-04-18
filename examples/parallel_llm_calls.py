"""Parallel LLM calls in a single tree.

Runs three LLM calls concurrently using Parallel, then prints all responses
collected on the shared blackboard.
"""

import asyncio
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable

from tinytasktree import JSON, Context, FileTraceStorageHandler, LLMModel, LLMProvider, Tree

# Requirements:
#   - LLM_BASE_URL and LLM_API_KEY set for your LLM service
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
PROVIDER = LLMProvider(base_url=LLM_BASE_URL or "", api_key=LLM_API_KEY)
MODEL_1 = LLMModel("qwen/qwen3.6-plus", provider=PROVIDER, extra_body={"reasoning": {"enabled": False}})
MODEL_2 = LLMModel("qwen/qwen3.5-35b-a3b", provider=PROVIDER, extra_body={"reasoning": {"enabled": False}})
MODEL_3 = LLMModel("qwen/qwen3.5-27b", provider=PROVIDER, extra_body={"reasoning": {"enabled": False}})


@dataclass
class Blackboard:
    prompt: str
    response_1: str = ""
    response_2: str = ""
    response_3: str = ""


def make_messages(b: Blackboard) -> list[JSON]:
    return [{"role": "user", "content": b.prompt}]


def make_messages_1(b: Blackboard) -> list[JSON]:
    return make_messages(b)


def make_messages_2(b: Blackboard) -> list[JSON]:
    return make_messages(b)


def make_messages_3(b: Blackboard) -> list[JSON]:
    return make_messages(b)


def write_response_1(b: Blackboard, data: str) -> None:
    b.response_1 = data.strip()


def write_response_2(b: Blackboard, data: str) -> None:
    b.response_2 = data.strip()


def write_response_3(b: Blackboard, data: str) -> None:
    b.response_3 = data.strip()


# fmt: off
tree = (
    Tree[Blackboard]("ParallelLLM")
    .Parallel(concurrency_limit=3)
    ._().Sequence()
    ._()._().LLM(MODEL_1, make_messages_1)
    ._()._().WriteBlackboard(write_response_1)
    ._().Sequence()
    ._()._().LLM(MODEL_2, make_messages_2)
    ._()._().WriteBlackboard(write_response_2)
    ._().Sequence()
    ._()._().LLM(MODEL_3, make_messages_3)
    ._()._().WriteBlackboard(write_response_3)
    .End()
)
# fmt: on


async def main() -> None:
    context = Context()
    blackboard = Blackboard(prompt="Give a one-sentence fun fact about space.")

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("Result:", result)
    print("Response 1:", blackboard.response_1)
    print("Response 2:", blackboard.response_2)
    print("Response 3:", blackboard.response_3)

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:8000/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())

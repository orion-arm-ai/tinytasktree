"""Gather multiple LLM calls with per-call blackboards.

Creates 1-5 subtrees at runtime, runs them concurrently with Gather, and
returns a list of responses in the original order.
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
MODEL = LLMModel("qwen/qwen3.6-plus", provider=PROVIDER, llm_call_kwargs={"reasoning": {"enabled": False}})


@dataclass
class RootBlackboard:
    prompts: list[str]
    responses: list[str] | None = None


@dataclass
class SubBlackboard:
    prompt: str
    response: str = ""


def make_messages(b: SubBlackboard) -> list[JSON]:
    return [{"role": "user", "content": b.prompt}]


def write_response(b: SubBlackboard, data: str) -> None:
    b.response = data


# Subtree used by each gathered task
# fmt: off
subtree = (
    Tree[SubBlackboard]("LLMCall")
    .Sequence()
    ._().LLM(MODEL, make_messages)
    ._().WriteBlackboard(write_response)
    .End()
)
# fmt: on


def gather_params(b: RootBlackboard):
    num_calls = min(5, max(1, len(b.prompts)))
    trees = [subtree] * num_calls
    blackboards = [SubBlackboard(prompt=p) for p in b.prompts[:num_calls]]
    return trees, blackboards


# fmt: off
tree = (
    Tree[RootBlackboard]("GatherLLM")
    .Sequence()
    ._().Gather(params_factory=gather_params, concurrency_limit=3)
    ._().WriteBlackboard("responses")
    .End()
)
# fmt: on


async def main() -> None:
    prompts = [
        "Give one fun fact about space.",
        "Give one fun fact about oceans.",
        "Give one fun fact about mountains.",
        "Give one fun fact about deserts.",
        "Give one fun fact about forests.",
    ]
    blackboard = RootBlackboard(prompts=prompts)
    context = Context()

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("Result:", result)
    print("Responses:")
    for i, text in enumerate(blackboard.responses or []):
        print(f"  {i + 1}. {text}")

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:8000/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())

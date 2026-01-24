"""Gather multiple LLM calls with per-call blackboards.

Creates 1-5 subtrees at runtime, runs them concurrently with Gather, and
returns a list of responses in the original order.
"""

import asyncio
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable

from tinytasktree import JSON, Context, FileTraceStorageHandler, Tree

# Requirements:
#   - OPENROUTER_API_KEY set for OpenRouter access via LiteLLM


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
    ._().LLM("openrouter/openai/gpt-4.1-mini", make_messages)
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
    print("Trace URL:", f"http://127.0.0.1:5173/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())

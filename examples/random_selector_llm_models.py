"""RandomSelector among LLM models for the same prompt.

Each run picks a model at random (optionally weighted) and returns its response.
"""

import asyncio
import os
import sys
from dataclasses import dataclass

import litellm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable
from tinytasktree import JSON, Context, FileTraceStorageHandler, Tree

litellm.suppress_debug_info = True
litellm.set_verbose = False

# Requirements:
#   - OPENROUTER_API_KEY set for OpenRouter access via LiteLLM


@dataclass
class Blackboard:
    prompt: str
    response: str = ""


def make_messages(b: Blackboard) -> list[JSON]:
    return [{"role": "user", "content": b.prompt}]


def on_delta(b: Blackboard, fulltext: str, delta: str, finished: bool) -> None:
    print(delta, end="", flush=True)


def write_response(b: Blackboard, data: str) -> None:
    b.response = data


# fmt: off
tree = (
    Tree[Blackboard]("RandomSelectorLLM")
    .Sequence()
    ._().RandomSelector(weights=[0.4, 0.4, 0.2]) # sets weights=None for equal probability
    ._()._().LLM("openrouter/openai/gpt-oss-120b:free", make_messages, stream=True, stream_on_delta=on_delta, name="ModelA")
    ._()._().LLM("openrouter/google/gemma-3-27b-it:free", make_messages, stream=True, stream_on_delta=on_delta, name="ModelB")
    ._()._().LLM("openrouter/openai/gpt-4.1-mini", make_messages, stream=True, stream_on_delta=on_delta, name="ModelC")
    ._().WriteBlackboard(write_response)
    .End()
)
# fmt: on


async def main() -> None:
    context = Context()
    blackboard = Blackboard(prompt="Write a single-sentence greeting.")

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("\nResult:", result)
    print("Response:", blackboard.response)

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:5173/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())

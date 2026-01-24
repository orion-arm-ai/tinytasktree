"""Selector fallback between LLMs with timeout handling.

Tries a fast LLM under a timeout; if it fails, the Selector falls back to a
second LLM and writes the successful response to the blackboard.
"""

import asyncio
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable
import litellm

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
    Tree[Blackboard]("SelectorFallback")
    .Sequence()
    ._().Selector()
    ._()._().Timeout(3) # First attemp
    ._()._()._().LLM("openrouter/openai/gpt-oss-120b:free", make_messages, stream=True, stream_on_delta=on_delta, name="FirstAttempLLM")
    ._()._()._().WriteBlackboard(write_response)
    ._()._().LLM("openrouter/google/gemma-3-27b-it:free", make_messages, stream=True, stream_on_delta=on_delta, name="FallbackLLM")
    ._().WriteBlackboard(write_response)
    .End()
)
# fmt: on


async def main() -> None:
    context = Context()
    blackboard = Blackboard(prompt="Say hello in one short sentence.")

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("\nResult:", result)
    print("Response:", blackboard.response)

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:5173/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())

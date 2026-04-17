"""Parse JSON from an LLM response.

Calls an LLM that returns JSON text, parses it into a Python object using
ParseJSON, and prints the parsed result.
"""

import asyncio
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable

from tinytasktree import JSON, Context, FileTraceStorageHandler, Tree

# Requirements:
#   - OPENROUTER_BASE_URL and OPENROUTER_API_KEY set for OpenRouter access
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


@dataclass
class Blackboard:
    prompt: str
    parsed: JSON | None = None


def make_messages(b: Blackboard) -> list[JSON]:
    return [{"role": "user", "content": b.prompt}]


# fmt: off
tree = (
    Tree[Blackboard]("ParseJSON")
    .Sequence()
    ._().LLM("openai/gpt-4.1-mini", make_messages, base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
    ._().ParseJSON(dst="parsed")
    .End()
)
# fmt: on


async def main() -> None:
    blackboard = Blackboard(
        prompt=("Return a JSON object with keys: name (string), year (number), tags (array of strings).")
    )
    context = Context()

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("Result:", result)
    print("Parsed:", blackboard.parsed)

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:5173/{trace_id}")

if __name__ == "__main__":
    asyncio.run(main())

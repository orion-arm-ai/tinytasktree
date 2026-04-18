"""Parse JSON with optional json_repair installed.

Demonstrates default ParseJSON behavior when the optional json_repair package
is installed. No custom json_loader is needed.
"""

import asyncio
import importlib.util
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable

from tinytasktree import JSON, Context, FileTraceStorageHandler, Tree


@dataclass
class Blackboard:
    raw_json: str
    parsed: JSON | None = None


# fmt: off
tree = (
    Tree[Blackboard]("ParseJSONWithJsonRepair")
    .Sequence()
    ._().ParseJSON(src="raw_json", dst="parsed")
    .End()
)
# fmt: on

storage = FileTraceStorageHandler(".traces")


async def main() -> None:
    if importlib.util.find_spec("json_repair") is None:
        raise SystemExit("Install the optional package `json_repair` to run this example.")

    context = Context()
    blackboard = Blackboard(raw_json='{"name": "tinytasktree", "year": 2026')

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("Result:", result)
    print("Parsed:", blackboard.parsed)
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:8000/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())

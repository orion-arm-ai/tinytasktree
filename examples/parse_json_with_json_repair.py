"""Parse JSON with an explicit json_repair loader.

Demonstrates how to keep tinytasktree free of a json_repair dependency while
still using json_repair in application code when tolerant parsing is needed.
"""

import asyncio
import os
import sys
from dataclasses import dataclass

import json_repair

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable

from tinytasktree import JSON, Context, FileTraceStorageHandler, Tree


@dataclass
class Blackboard:
    raw_json: str
    parsed: JSON | None = None


def repairing_json_loader(s: str) -> JSON | None:
    s = s.strip()
    if s.startswith("```json"):
        s = s[len("```json") :]
    elif s.startswith("```"):
        s = s[len("```") :]
    if s.endswith("```"):
        s = s[: -len("```")]
    return json_repair.loads(s)


# fmt: off
tree = (
    Tree[Blackboard]("ParseJSONWithJsonRepair")
    .Sequence()
    ._().ParseJSON(src="raw_json", dst="parsed", json_loader=repairing_json_loader)
    .End()
)
# fmt: on

storage = FileTraceStorageHandler(".traces")


async def main() -> None:
    context = Context()
    blackboard = Blackboard(raw_json='{"name": "tinytasktree", "year": 2026')

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("Result:", result)
    print("Parsed:", blackboard.parsed)
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:5173/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())

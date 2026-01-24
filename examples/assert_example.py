"""Assert example with a blackboard condition.

Checks a condition using Assert; if it fails, the Sequence stops and the next
node is skipped.
"""

import asyncio
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable

from tinytasktree import Context, FileTraceStorageHandler, Tree


@dataclass
class Blackboard:
    should_run: bool
    message: str = ""


def set_message(b: Blackboard) -> str:
    return "This runs only if assertion passes."


def write_message(b: Blackboard, data: str) -> None:
    b.message = data


# fmt: off
tree = (
    Tree[Blackboard]("AssertExample")
    .Sequence()
    ._().Assert(lambda b: b.should_run)
    ._().Function(set_message)
    ._().WriteBlackboard(write_message)
    .End()
)
# fmt: on


async def main() -> None:
    context = Context()

    blackboard_ok = Blackboard(should_run=True)
    async with context.using_blackboard(blackboard_ok):
        result_ok = await tree(context)
    print("Result (pass):", result_ok)
    print("Message (pass):", blackboard_ok.message)

    blackboard_fail = Blackboard(should_run=False)
    async with context.using_blackboard(blackboard_fail):
        result_fail = await tree(context)
    print("Result (fail):", result_fail)
    print("Message (fail):", blackboard_fail.message)

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:5173/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())

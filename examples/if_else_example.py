"""If/Else branching with a simple condition.

Uses If/Else to choose between two branches based on a boolean on the blackboard,
then stores the chosen message.
"""

import asyncio
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable

from tinytasktree import Context, FileTraceStorageHandler, Result, Tree


@dataclass
class Blackboard:
    use_formal: bool
    message: str = ""


def set_formal(b: Blackboard) -> str:
    return "Hello. How may I assist you today?"


def set_casual(b: Blackboard) -> str:
    return "Hey! What can I help you with?"


def write_message(b: Blackboard, data: str) -> None:
    b.message = data


# fmt: off
tree = (
    Tree[Blackboard]("IfElse")
    .Sequence()
    ._().If(lambda b: b.use_formal)
    ._()._().Function(set_formal)
    ._()._().Else() # Note: `Else` is a child of `If`.
    ._()._()._().Function(set_casual)
    ._().WriteBlackboard(write_message)
    .End()
)
# fmt: on


async def main() -> None:
    blackboard = Blackboard(use_formal=False)
    context = Context()

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("Result:", result)
    print("Message:", blackboard.message)

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:5173/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())

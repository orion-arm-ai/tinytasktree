"""Subtree example with separate blackboard types.

Builds a root tree that calls a subtree using a derived blackboard, then writes
back the subtree result into the root blackboard.
"""

import asyncio
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable

from tinytasktree import Context, FileTraceStorageHandler, Tree


@dataclass
class RootBlackboard:
    input_text: str
    sub_result: str = ""


@dataclass
class SubBlackboard:
    text: str
    uppercased: str = ""


def to_sub_blackboard(b: RootBlackboard) -> SubBlackboard:
    return SubBlackboard(text=b.input_text)


def to_upper(b: SubBlackboard) -> str:
    return b.text.upper()


def write_upper(b: SubBlackboard, data: str) -> None:
    b.uppercased = data


def write_sub_result(b: RootBlackboard, data: str) -> None:
    b.sub_result = data


# fmt: off
subtree = (
    Tree[SubBlackboard]("Uppercase")
    .Sequence()
    ._().Function(to_upper)
    ._().WriteBlackboard(write_upper)
    .End()
)

tree = (
    Tree[RootBlackboard]("Root")
    .Sequence()
    ._().Subtree(subtree, to_sub_blackboard)
    ._().WriteBlackboard(write_sub_result)
    .End()
)
# fmt: on


async def main() -> None:
    blackboard = RootBlackboard(input_text="hello subtree")
    context = Context()

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("Result:", result)
    print("Subtree result:", blackboard.sub_result)

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:5173/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())

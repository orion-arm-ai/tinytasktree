"""Examples for ForceOk, ForceFail, Return, and Invert decorators."""

import asyncio
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable

from tinytasktree import Context, FileTraceStorageHandler, Tree


@dataclass
class Blackboard:
    value: int = 1


def bump_value(b: Blackboard) -> int:
    b.value += 1
    return b.value


def as_payload(b: Blackboard) -> dict[str, int]:
    return {"value": b.value}


# fmt: off
tree_force_ok = (
    Tree[Blackboard]("ForceOkExample")
    .ForceOk()
    ._().Failure()
    .End()
)

tree_force_fail = (
    Tree[Blackboard]("ForceFailExample")
    .ForceFail(lambda b: f"forced fail (value={b.value})")
    ._().Function(bump_value)
    .End()
)

tree_invert = (
    Tree[Blackboard]("InvertExample")
    .Invert()
    ._().Failure()
    .End()
)

tree_return = (
    Tree[Blackboard]("ReturnExample")
    .Return(as_payload)
    ._().Function(bump_value)
    .End()
)
# fmt: on


async def run_one(context: Context, tree: Tree[Blackboard], blackboard: Blackboard) -> None:
    async with context.using_blackboard(blackboard):
        result = await tree(context)
    print(f"{tree.name}: {result}, blackboard.value={blackboard.value}")


async def main() -> None:
    context = Context()

    await run_one(context, tree_force_ok, Blackboard())
    await run_one(context, tree_force_fail, Blackboard())
    await run_one(context, tree_invert, Blackboard())
    await run_one(context, tree_return, Blackboard())

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:5173/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())

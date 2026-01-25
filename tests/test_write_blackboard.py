"""WriteBlackboard node behavior tests.

Steps:
- Write to blackboard via attribute name after a preceding Function.
- Write to blackboard via setter function after a preceding Function.
Expectations:
- WriteBlackboard returns OK with the last_result data.
- Blackboard receives the data via both attr and function paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import tinytasktree


@dataclass
class Blackboard:
    attr_value: str | None = None
    func_value: str | None = None


def set_func_value(b: Blackboard, data: str) -> None:
    b.func_value = data


async def test_write_blackboard_attr_and_result():
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("WriteAttr")
        .Sequence()
        ._().Function(lambda: "attr")
        ._().WriteBlackboard("attr_value")
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "attr"
    assert blackboard.attr_value == "attr"


async def test_write_blackboard_func_and_result():
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("WriteFunc")
        .Sequence()
        ._().Function(lambda: "func")
        ._().WriteBlackboard(set_func_value)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "func"
    assert blackboard.func_value == "func"

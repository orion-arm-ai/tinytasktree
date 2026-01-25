"""Decorator behavior tests for ForceOk/ForceFail/Return/Invert.

Steps:
- ForceOk wraps a failing child and returns OK with child data or factory data.
- ForceFail wraps a succeeding child and returns FAIL with child data or factory data.
- Return preserves child status but replaces data.
- Invert flips the child status and keeps data.
Expectations:
- Each decorator returns the expected status and data.
"""

from __future__ import annotations

from dataclasses import dataclass

import tinytasktree


@dataclass
class Blackboard:
    value: str = ""


async def test_force_ok_and_force_fail():
    def make_value(b: Blackboard) -> str:
        return b.value

    # fmt: off
    tree_force_ok = (
        tinytasktree.Tree[Blackboard]("ForceOk")
        .ForceOk()
        ._().Failure()
        .End()
    )
    tree_force_ok_factory = (
        tinytasktree.Tree[Blackboard]("ForceOkFactory")
        .ForceOk(make_value)
        ._().Failure()
        .End()
    )
    tree_force_fail = (
        tinytasktree.Tree[Blackboard]("ForceFail")
        .ForceFail()
        ._().Function(lambda: "child")
        .End()
    )
    tree_force_fail_factory = (
        tinytasktree.Tree[Blackboard]("ForceFailFactory")
        .ForceFail(make_value)
        ._().Function(lambda: "child")
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard(value="bb")
    async with context.using_blackboard(blackboard):
        result = await tree_force_ok(context)
    assert result.is_ok()
    assert result.data is None

    context = tinytasktree.Context()
    blackboard = Blackboard(value="bb")
    async with context.using_blackboard(blackboard):
        result = await tree_force_ok_factory(context)
    assert result.is_ok()
    assert result.data == "bb"

    context = tinytasktree.Context()
    blackboard = Blackboard(value="bb")
    async with context.using_blackboard(blackboard):
        result = await tree_force_fail(context)
    assert not result.is_ok()
    assert result.data == "child"

    context = tinytasktree.Context()
    blackboard = Blackboard(value="bb")
    async with context.using_blackboard(blackboard):
        result = await tree_force_fail_factory(context)
    assert not result.is_ok()
    assert result.data == "bb"


async def test_return_and_invert():
    def make_value(b: Blackboard) -> str:
        return f"ret:{b.value}"

    # fmt: off
    tree_return_ok = (
        tinytasktree.Tree[Blackboard]("ReturnOk")
        .Return(make_value)
        ._().Function(lambda: "child")
        .End()
    )
    tree_return_fail = (
        tinytasktree.Tree[Blackboard]("ReturnFail")
        .Return(make_value)
        ._().Failure()
        .End()
    )
    tree_invert_fail = (
        tinytasktree.Tree[Blackboard]("Invert")
        .Invert()
        ._().Failure()
        .End()
    )
    tree_invert_ok = (
        tinytasktree.Tree[Blackboard]("InvertOk")
        .Invert()
        ._().Function(lambda: "ok")
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard(value="v")
    async with context.using_blackboard(blackboard):
        result = await tree_return_ok(context)
    assert result.is_ok()
    assert result.data == "ret:v"

    context = tinytasktree.Context()
    blackboard = Blackboard(value="v")
    async with context.using_blackboard(blackboard):
        result = await tree_return_fail(context)
    assert not result.is_ok()
    assert result.data == "ret:v"

    context = tinytasktree.Context()
    blackboard = Blackboard(value="v")
    async with context.using_blackboard(blackboard):
        result = await tree_invert_fail(context)
    assert result.is_ok()
    assert result.data is None

    context = tinytasktree.Context()
    blackboard = Blackboard(value="v")
    async with context.using_blackboard(blackboard):
        result = await tree_invert_ok(context)
    assert not result.is_ok()
    assert result.data == "ok"

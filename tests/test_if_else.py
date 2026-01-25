"""If/Else node behavior tests.

Steps:
- If with true condition executes the then-branch.
- If with false condition executes the else branch.
- If with false condition and no else returns OK(None).
- If with attr-based condition uses blackboard boolean.
- Else used outside If raises TasktreeProgrammingError.
Expectations:
- Correct branch executes and result/data reflect the branch.
- Missing else yields OK(None) without error.
- Misplaced Else raises programming error.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import tinytasktree


@dataclass
class Blackboard:
    flag: bool = False
    seen: list[str] | None = None


def mark(b: Blackboard, value: str) -> str:
    if b.seen is None:
        b.seen = []
    b.seen.append(value)
    return value


async def test_if_true_then_branch():
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("IfTrue")
        .If(lambda: True)
        ._().Function(lambda b: mark(b, "then"))
        ._().Else()
        ._()._().Function(lambda b: mark(b, "else"))
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "then"
    assert blackboard.seen == ["then"]


async def test_if_false_else_branch():
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("IfFalse")
        .If(lambda: False)
        ._().Function(lambda b: mark(b, "then"))
        ._().Else()
        ._()._().Function(lambda b: mark(b, "else"))
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "else"
    assert blackboard.seen == ["else"]


async def test_if_false_without_else():
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("IfNoElse")
        .If(lambda: False)
        ._().Function(lambda b: mark(b, "then"))
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data is None
    assert blackboard.seen is None


async def test_if_condition_from_blackboard_attr():
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("IfAttr")
        .If("flag")
        ._().Function(lambda b: mark(b, "then"))
        ._().Else()
        ._()._().Function(lambda b: mark(b, "else"))
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard(flag=True)
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "then"
    assert blackboard.seen == ["then"]


async def test_else_must_be_child_of_if():
    # fmt: off
    with pytest.raises(tinytasktree.TasktreeProgrammingError):
        (
            tinytasktree.Tree[Blackboard]("ElseInvalid")
            .Sequence()
            ._().Else()
            ._()._().Function(lambda: None)
            .End()
        )
    # fmt: on

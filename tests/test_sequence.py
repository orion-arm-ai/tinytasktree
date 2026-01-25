"""Sequence node behavior tests.

Steps:
- Build a Sequence with all succeeding children and run it.
- Build a Sequence with a failing child after a success and run it.
Expectations:
- When all children succeed, Sequence returns OK and last result data is kept.
- When any child fails, Sequence returns FAIL and stops before later children.
"""

from __future__ import annotations

from dataclasses import dataclass

import tinytasktree


@dataclass
class Blackboard:
    seen: list[str]


async def test_sequence_all_ok():
    def push_a(b: Blackboard) -> str:
        b.seen.append("a")
        return "a"

    def push_b(b: Blackboard) -> str:
        b.seen.append("b")
        return "b"

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("SeqAllOk")
        .Sequence()
        ._().Function(push_a)
        ._().Function(push_b)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard(seen=[])
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "b"
    assert blackboard.seen == ["a", "b"]


async def test_sequence_any_failure():
    def push_a(b: Blackboard) -> str:
        b.seen.append("a")
        return "a"

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("SeqAnyFail")
        .Sequence()
        ._().Function(push_a)
        ._().Failure()
        ._().Function(lambda b: b.seen.append("c"))
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard(seen=[])
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert not result.is_ok()
    assert result.data == "a"
    assert blackboard.seen == ["a"]

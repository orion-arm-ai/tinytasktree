"""Selector node behavior tests.

Steps:
- Build a Selector with all failing children and run it.
- Build a Selector with a failing child followed by a succeeding child and run it.
Expectations:
- When all children fail, Selector returns FAIL.
- When any child succeeds, Selector returns OK and later children are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import tinytasktree


@dataclass
class Blackboard:
    chosen: str | None = None


async def test_selector_all_failures():
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("SelectorAllFail")
        .Selector()
        ._().Failure()
        ._().Failure()
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert not result.is_ok()
    assert blackboard.chosen is None


async def test_selector_any_success():
    def pick_ok(b: Blackboard) -> str:
        b.chosen = "ok"
        return "ok"

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("SelectorAnyOk")
        .Selector()
        ._().Failure()
        ._().Function(pick_ok)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert blackboard.chosen == "ok"

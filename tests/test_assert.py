"""Assert node behavior tests.

Steps:
- Assert a true condition via function.
- Assert a false condition via function.
- Assert that raised AssertionError is treated as failure.
Expectations:
- True condition returns OK(True).
- False condition returns FAIL with no prior success data in Sequence.
- AssertionError results in FAIL with no prior success data in Sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import tinytasktree


@dataclass
class Blackboard:
    flag: bool = False


async def test_assert_true_and_false():
    # fmt: off
    tree_true = (
        tinytasktree.Tree[Blackboard]("AssertTrue")
        .Sequence()
        ._().Assert(lambda: True)
        .End()
    )
    tree_false = (
        tinytasktree.Tree[Blackboard]("AssertFalse")
        .Sequence()
        ._().Assert(lambda: False)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree_true(context)
    assert result.is_ok()
    assert result.data is True

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree_false(context)
    assert not result.is_ok()
    assert result.data is None


async def test_assert_raises_assertion_error():
    def boom():
        raise AssertionError("nope")

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("AssertRaise")
        .Sequence()
        ._().Assert(boom)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert not result.is_ok()
    assert result.data is None

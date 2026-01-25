"""While node behavior tests.

Steps:
- Run a While loop with a condition that stops after a few iterations.
- Run a While loop where the child fails mid-way.
- Run a While loop that immediately stops.
Expectations:
- Loop runs until condition is false and returns last OK result.
- Failure inside loop stops iteration and returns last OK result.
- Immediate stop returns FAIL(None).
"""

from __future__ import annotations

from dataclasses import dataclass

import tinytasktree


@dataclass
class Blackboard:
    count: int = 0


def inc(b: Blackboard) -> int:
    b.count += 1
    return b.count


def fail_on_two(b: Blackboard) -> tinytasktree.Result:
    b.count += 1
    if b.count == 2:
        return tinytasktree.Result.FAIL(None)
    return tinytasktree.Result.OK(b.count)


async def test_while_runs_until_condition_false():
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("WhileOk")
        .While(lambda b: b.count < 3)
        ._().Function(inc)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard(count=0)
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == 3
    assert blackboard.count == 3


async def test_while_stops_on_child_failure():
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("WhileFail")
        .While(lambda b: b.count < 5)
        ._().Function(fail_on_two)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard(count=0)
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == 1
    assert blackboard.count == 2


async def test_while_immediate_stop():
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("WhileStop")
        .While(lambda b: False)
        ._().Function(inc)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard(count=0)
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert not result.is_ok()
    assert result.data is None
    assert blackboard.count == 0

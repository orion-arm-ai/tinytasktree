"""While node max_loop_times behavior tests.

Steps:
- Run a While loop with a condition that never becomes false.
- Set max_loop_times to a small number.
Expectations:
- Loop stops after max_loop_times and returns last OK result.
- Counter reflects the max loop limit.
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


async def test_while_hits_max_loop_times():
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("WhileMaxLoop")
        .While(lambda b: True, max_loop_times=3)
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

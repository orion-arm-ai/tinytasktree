"""RandomSelector node behavior tests.

Steps:
- Seed RNG and compute expected selection order for given weights.
- Build a RandomSelector with mixed failing and succeeding children.
- Run the tree and record which children executed.
Expectations:
- Execution order matches the weighted shuffle for the seeded RNG.
- Selector stops at the first OK child and returns its data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import tinytasktree


@dataclass
class Blackboard:
    visited: list[int]


def _make_child(idx: int, ok: bool):
    def run(b: Blackboard):
        b.visited.append(idx)
        if ok:
            return f"ok-{idx}"
        return tinytasktree.Result.FAIL(None)

    return run


async def test_random_selector_weighted_order_and_stop():
    weights = [1.0, 2.0, 3.0]

    random.seed(123)
    expected_order = tinytasktree._weighted_shuffle([0, 1, 2], weights=weights)
    expected_first_ok = next(i for i in expected_order if i in {1, 2})
    expected_visited = expected_order[: expected_order.index(expected_first_ok) + 1]

    random.seed(123)
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("RandomSelector")
        .RandomSelector(weights=weights)
        ._().Function(_make_child(0, ok=False))
        ._().Function(_make_child(1, ok=True))
        ._().Function(_make_child(2, ok=True))
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard(visited=[])
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == f"ok-{expected_first_ok}"
    assert blackboard.visited == expected_visited
    assert len(set(blackboard.visited)) == len(blackboard.visited)

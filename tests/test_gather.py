"""Gather node behavior tests.

Steps:
- Gather two subtrees with matching blackboards.
- Verify data list order matches subtree order.
- Verify failure in any child yields FAIL status.
- Verify mismatch in trees/blackboards raises a programming error.
Expectations:
- Gather returns OK with data list when all children succeed.
- Gather returns FAIL when any child fails, but still returns data list.
- Mismatched params raise TasktreeProgrammingError to fail fast.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import tinytasktree


@dataclass
class ParentBoard:
    base: int


@dataclass
class ChildBoard:
    value: int


def make_child_tree_ok():
    # fmt: off
    return (
        tinytasktree.Tree[ChildBoard]("ChildOk")
        .Sequence()
        ._().Function(lambda b: b.value)
        .End()
    )
    # fmt: on


def make_child_tree_fail():
    # fmt: off
    return (
        tinytasktree.Tree[ChildBoard]("ChildFail")
        .Sequence()
        ._().Failure()
        .End()
    )
    # fmt: on


async def test_gather_all_ok():
    def params_factory(_: ParentBoard):
        trees = [make_child_tree_ok(), make_child_tree_ok()]
        boards = [ChildBoard(1), ChildBoard(2)]
        return trees, boards

    # fmt: off
    tree = (
        tinytasktree.Tree[ParentBoard]("GatherAllOk")
        .Gather(params_factory)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = ParentBoard(base=0)
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == [1, 2]


async def test_gather_any_fail():
    def params_factory(_: ParentBoard):
        trees = [make_child_tree_ok(), make_child_tree_fail()]
        boards = [ChildBoard(1), ChildBoard(2)]
        return trees, boards

    # fmt: off
    tree = (
        tinytasktree.Tree[ParentBoard]("GatherAnyFail")
        .Gather(params_factory)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = ParentBoard(base=0)
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert not result.is_ok()
    assert result.data == [1, None]


async def test_gather_params_mismatch():
    def params_factory(_: ParentBoard):
        trees = [make_child_tree_ok()]
        boards = [ChildBoard(1), ChildBoard(2)]
        return trees, boards

    # fmt: off
    tree = (
        tinytasktree.Tree[ParentBoard]("GatherMismatch")
        .Gather(params_factory)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = ParentBoard(base=0)
    async with context.using_blackboard(blackboard):
        with pytest.raises(tinytasktree.TasktreeProgrammingError):
            await tree(context)

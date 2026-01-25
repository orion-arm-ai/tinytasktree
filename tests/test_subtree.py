"""Subtree node behavior tests.

Steps:
- Build a subtree that mutates a child blackboard and returns a value.
- Attach it in a parent tree without a custom blackboard factory.
- Attach it in a parent tree with a custom blackboard factory.
Expectations:
- Without a factory, the parent blackboard is used and mutated.
- With a factory, the child blackboard is used and parent remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import tinytasktree


@dataclass
class ParentBoard:
    value: int = 0


@dataclass
class ChildBoard:
    value: int = 0


def add_one_parent(b: ParentBoard) -> int:
    b.value += 1
    return b.value


def add_one_child(b: ChildBoard) -> int:
    b.value += 1
    return b.value


async def test_subtree_without_custom_blackboard():
    # fmt: off
    subtree = (
        tinytasktree.Tree[ParentBoard]("SubNoFactory")
        .Sequence()
        ._().Function(add_one_parent)
        .End()
    )
    tree = (
        tinytasktree.Tree[ParentBoard]("ParentNoFactory")
        .Sequence()
        ._().Subtree(subtree)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = ParentBoard(value=0)
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert blackboard.value == 1


async def test_subtree_with_custom_blackboard():
    def make_child(b: ParentBoard) -> ChildBoard:
        return ChildBoard(value=b.value)

    # fmt: off
    subtree = (
        tinytasktree.Tree[ChildBoard]("SubWithFactory")
        .Sequence()
        ._().Function(add_one_child)
        .End()
    )
    tree = (
        tinytasktree.Tree[ParentBoard]("ParentWithFactory")
        .Sequence()
        ._().Subtree(subtree, make_child)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = ParentBoard(value=0)
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert blackboard.value == 0

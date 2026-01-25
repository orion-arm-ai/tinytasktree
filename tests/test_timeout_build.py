"""Timeout build-time validation tests.

Steps:
- Build a Timeout with no child.
- Build a Timeout with too many children.
Expectations:
- Build errors raise TasktreeProgrammingError at build time.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import tinytasktree


@dataclass
class Blackboard:
    value: str = ""


async def test_timeout_with_no_child_fails():
    # fmt: off
    with pytest.raises(tinytasktree.TasktreeProgrammingError):
        (
            tinytasktree.Tree[Blackboard]("TimeoutNoChild")
            .Timeout(0.1)
            .End()
        )
    # fmt: on


async def test_timeout_with_too_many_children_fails():
    # fmt: off
    with pytest.raises(tinytasktree.TasktreeProgrammingError):
        (
            tinytasktree.Tree[Blackboard]("TimeoutTooMany")
            .Timeout(0.1)
            ._().Function(lambda: "a")
            ._().Function(lambda: "b")
            ._().Function(lambda: "c")
            .End()
        )
    # fmt: on

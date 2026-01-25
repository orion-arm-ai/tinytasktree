"""Parallel node build-time validation tests.

Steps:
- Attempt to build a Parallel node with concurrency_limit=0.
- Assert that build raises a TasktreeProgrammingError.
Expectations:
- Parallel.OnBuildEnd rejects non-positive concurrency limits.
"""

from __future__ import annotations

import pytest

import tinytasktree


def test_parallel_concurrency_limit_must_be_positive():
    with pytest.raises(tinytasktree.TasktreeProgrammingError):
        # fmt: off
        (
            tinytasktree.Tree[object]("ParallelInvalid")
            .Parallel(concurrency_limit=0)
            ._().Failure()
            .End()
        )
        # fmt: on

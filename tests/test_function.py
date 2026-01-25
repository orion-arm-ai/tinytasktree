"""Function node behavior tests.

Steps:
- Build Function nodes covering all supported call signatures (sync/async, 0/1/2 params, Any/Result).
- Execute each tree and assert status/data matches expectations.
- Execute a Function that raises an exception and assert FAIL(None).
Expectations:
- All supported function forms run and return expected results.
- Exceptions are caught and converted to FAIL(None).
"""

from __future__ import annotations

from dataclasses import dataclass

import tinytasktree


@dataclass
class Blackboard:
    value: int = 0


def _build_tree(func):
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("FunctionTree")
        .Sequence()
        ._().Function(func)
        .End()
    )
    # fmt: on
    return tree


async def test_function_all_supported_forms():
    def sync0_any():
        return "v0"

    def sync0_result():
        return tinytasktree.Result.OK("r0")

    def sync1_any(b: Blackboard):
        b.value += 1
        return b.value

    def sync1_result(b: Blackboard):
        b.value += 1
        return tinytasktree.Result.OK(b.value)

    def sync2_any(b: Blackboard, tracer: tinytasktree.Tracer):
        b.value += 1
        return b.value

    def sync2_result(b: Blackboard, tracer: tinytasktree.Tracer):
        b.value += 1
        return tinytasktree.Result.OK(b.value)

    async def async0_any():
        return "av0"

    async def async0_result():
        return tinytasktree.Result.OK("ar0")

    async def async1_any(b: Blackboard):
        b.value += 1
        return b.value

    async def async1_result(b: Blackboard):
        b.value += 1
        return tinytasktree.Result.OK(b.value)

    async def async2_any(b: Blackboard, tracer: tinytasktree.Tracer):
        b.value += 1
        return b.value

    async def async2_result(b: Blackboard, tracer: tinytasktree.Tracer):
        b.value += 1
        return tinytasktree.Result.OK(b.value)

    cases = [
        (sync0_any, "v0"),
        (sync0_result, "r0"),
        (sync1_any, 1),
        (sync1_result, 1),
        (sync2_any, 1),
        (sync2_result, 1),
        (async0_any, "av0"),
        (async0_result, "ar0"),
        (async1_any, 1),
        (async1_result, 1),
        (async2_any, 1),
        (async2_result, 1),
    ]

    for func, expected in cases:
        tree = _build_tree(func)
        context = tinytasktree.Context()
        blackboard = Blackboard(value=0)
        async with context.using_blackboard(blackboard):
            result = await tree(context)
        assert result.is_ok()
        assert result.data == expected


async def test_function_exception_results_in_fail():
    def boom():
        raise ValueError("boom")

    tree = _build_tree(boom)
    context = tinytasktree.Context()
    blackboard = Blackboard(value=0)
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert not result.is_ok()
    assert result.data is None

"""Spawned task finish hook tests.

Steps:
- Register a global spawned-task-finish hook that records calls.
- Run Parallel, Gather, and Terminable nodes that spawn child tasks.
- Verify the hook fires once per spawned child task.
Expectations:
- Parallel triggers the hook for each child.
- Gather triggers the hook for each gathered child.
- Terminable triggers the hook for its main child task.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import tinytasktree


@dataclass
class Blackboard:
    job_id: str


@dataclass
class ChildBoard:
    name: str


def _reset_hooks():
    hooks = list(tinytasktree._GLOBAL_HOOK_AFTER_SPAWNED_TASK_FINISH)
    tinytasktree._GLOBAL_HOOK_AFTER_SPAWNED_TASK_FINISH.clear()
    return hooks


def _restore_hooks(hooks):
    tinytasktree._GLOBAL_HOOK_AFTER_SPAWNED_TASK_FINISH.clear()
    tinytasktree._GLOBAL_HOOK_AFTER_SPAWNED_TASK_FINISH.extend(hooks)


async def test_spawned_task_finish_hook_parallel_gather_terminable(redis_url):
    calls: list[str] = []
    hooks = _reset_hooks()

    async def hook(context: tinytasktree.Context, tracer: tinytasktree.Tracer, result: tinytasktree.Result) -> None:
        calls.append(tracer.kind or tracer.name)

    tinytasktree.register_global_hook_after_spawned_task_finish(hook)

    try:
        # fmt: off
        parallel_tree = (
            tinytasktree.Tree[Blackboard]("HookParallel")
            .Parallel()
            ._().Function(lambda: "a")
            ._().Function(lambda: "b")
            .End()
        )
        # fmt: on

        calls.clear()
        context = tinytasktree.Context()
        blackboard = Blackboard(job_id=str(uuid.uuid4()))
        async with context.using_blackboard(blackboard):
            result = await parallel_tree(context)
        assert result.is_ok()
        assert len(calls) == 2

        def gather_params(_: Blackboard):
            # fmt: off
            child_tree = (
                tinytasktree.Tree[ChildBoard]("Child")
                .Function(lambda b: b.name)
                .End()
            )
            # fmt: on
            trees = [child_tree, child_tree]
            boards = [ChildBoard(name="x"), ChildBoard(name="y")]
            return trees, boards

        # fmt: off
        gather_tree = (
            tinytasktree.Tree[Blackboard]("HookGather")
            .Gather(gather_params)
            .End()
        )
        # fmt: on

        calls.clear()
        context = tinytasktree.Context()
        blackboard = Blackboard(job_id=str(uuid.uuid4()))
        async with context.using_blackboard(blackboard):
            result = await gather_tree(context)
        assert result.is_ok()
        assert len(calls) == 2

        tinytasktree.set_default_global_redis_client(redis_url)

        # fmt: off
        terminable_tree = (
            tinytasktree.Tree[Blackboard]("HookTerminable")
            .Terminable(lambda b: f"test:terminate:{b.job_id}", monitor_interval_ms=10)
            ._().Function(lambda: "done")
            .End()
        )
        # fmt: on

        calls.clear()
        context = tinytasktree.Context()
        blackboard = Blackboard(job_id=str(uuid.uuid4()))
        async with context.using_blackboard(blackboard):
            result = await terminable_tree(context)
        assert result.is_ok()
        assert len(calls) == 1
    finally:
        _restore_hooks(hooks)

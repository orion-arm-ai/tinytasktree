"""While + Gather to grow a topic tree.

Builds a small topic tree up to 3 levels. Each iteration pops the current queue
of topics, uses Gather to expand them with LLM calls, and enqueues the next level.
"""

import asyncio
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable

from tinytasktree import JSON, Context, FileTraceStorageHandler, Tree

# Requirements:
#   - OPENROUTER_API_KEY set for OpenRouter access via LiteLLM


@dataclass
class RootBlackboard:
    root_topic: str
    queue: list[tuple[str, int]] = field(default_factory=list)
    current_batch: list[tuple[str, int]] = field(default_factory=list)
    tree: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class SubBlackboard:
    topic: str
    level: int
    children: list[str] = field(default_factory=list)


def batch_from_queue(b: RootBlackboard) -> None:
    b.current_batch = b.queue
    b.queue = []


def make_messages(b: SubBlackboard) -> list[JSON]:
    return [
        {
            "role": "user",
            "content": (f"Return a JSON array of up to 3 short subtopics for the topic: {b.topic}. Return only JSON."),
        }
    ]


def store_children(b: SubBlackboard, data: JSON) -> None:
    if isinstance(data, list):
        b.children = [str(x) for x in data][:3]
    else:
        b.children = []


def return_expansion(b: SubBlackboard):
    return {"topic": b.topic, "level": b.level, "children": b.children}


# Subtree used for each topic expansion
# fmt: off
expand_subtree = (
    Tree[SubBlackboard]("ExpandTopic")
    .Return(return_expansion)
    ._().Sequence()
    ._()._().LLM("openrouter/openai/gpt-4.1-mini", make_messages)
    ._()._().ParseJSON(dst=store_children)
    .End()
)
# fmt: on


def gather_params(b: RootBlackboard):
    trees = [expand_subtree] * len(b.current_batch)
    blackboards = [SubBlackboard(topic=t, level=level) for t, level in b.current_batch]
    return trees, blackboards


def apply_expansions(b: RootBlackboard, data_list: list[dict]) -> None:
    max_level = 3
    for item in data_list:
        topic = item.get("topic", "")
        level = int(item.get("level", 0))
        children = item.get("children", []) or []
        b.tree[topic] = children
        if level < max_level:
            for child in children[:3]:
                b.queue.append((child, level + 1))


# fmt: off
tree = (
    Tree[RootBlackboard]("TopicTree")
    .Sequence()
    ._().While(lambda b: len(b.queue) > 0, max_loop_times=100)
    ._()._().Sequence()
    ._()._()._().Function(batch_from_queue)
    ._()._()._().Gather(params_factory=gather_params, concurrency_limit=3)
    ._()._()._().WriteBlackboard(apply_expansions)
    .End()
)
# fmt: on


async def main() -> None:
    root = RootBlackboard(root_topic="Artificial Intelligence")
    root.queue = [(root.root_topic, 1)]
    context = Context()

    async with context.using_blackboard(root):
        result = await tree(context)

    print("Result:", result)
    print("Tree:")
    print_tree(root.root_topic, root.tree, max_depth=3)
    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:5173/{trace_id}")


def print_tree(root: str, tree: dict[str, list[str]], max_depth: int = 3) -> None:
    def _walk(node: str, prefix: str, depth: int, is_last: bool) -> None:
        connector = "`-- " if is_last else "|-- "
        if depth == 0:
            print(node)
        else:
            print(prefix + connector + node)
        if depth >= max_depth:
            return
        children = tree.get(node, [])
        if not children:
            return
        next_prefix = prefix + ("    " if is_last else "|   ")
        for i, child in enumerate(children):
            _walk(child, next_prefix, depth + 1, i == len(children) - 1)

    _walk(root, "", 0, True)


if __name__ == "__main__":
    asyncio.run(main())

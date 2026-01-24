"""Complex tree example.

Demonstrates Parallel, If/Else, Retry, RandomSelector, ParseJSON, and Log nodes
in a single tree.
"""

import asyncio
import os
import random
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable

from tinytasktree import JSON, Context, FileTraceStorageHandler, Result, Tree

# Requirements:
#   - OPENROUTER_API_KEY set for OpenRouter access via LiteLLM


@dataclass
class Blackboard:
    prompt: str
    use_formal: bool = False
    style: str = ""
    summary: str = ""
    title: str = ""
    expected: int = 0
    guess_json: JSON | None = None
    hint: str = ""


def init_problem(b: Blackboard) -> None:
    b.use_formal = random.choice([True, False])
    b.expected = random.randint(1, 5)
    b.hint = ""


def make_summary_messages(b: Blackboard) -> list[JSON]:
    return [{"role": "user", "content": f"Summarize in 1 sentence: {b.prompt}"}]


def make_title_messages(b: Blackboard) -> list[JSON]:
    return [{"role": "user", "content": f"Give a short title: {b.prompt}"}]


def set_formal(_: Blackboard) -> str:
    return "formal"


def set_casual(_: Blackboard) -> str:
    return "casual"


def make_guess_messages(b: Blackboard) -> list[JSON]:
    hint = f" Hint: {b.hint}." if b.hint else ""
    prompt = f'Guess the secret number (1-5). Return only JSON with no extra text. Format: {{"guess": number}}.{hint}'
    return [{"role": "user", "content": prompt}]


def validate_guess(b: Blackboard) -> Result:
    if not b.guess_json or "guess" not in b.guess_json:
        b.hint = "invalid JSON"
        return Result.FAIL(None)
    try:
        value = int(b.guess_json["guess"])
    except (TypeError, ValueError):
        b.hint = "invalid JSON"
        return Result.FAIL(None)
    if value == b.expected:
        b.hint = ""
        return Result.OK(b.guess_json)
    b.hint = "too small" if value < b.expected else "too large"
    return Result.FAIL(b.guess_json)


# fmt: off
summary_tree = (
    Tree[Blackboard]("SummaryTree")
    .Sequence()
    ._().Log("Summary branch")
    ._().LLM("openrouter/openai/gpt-5-nano", make_summary_messages)
    ._().WriteBlackboard("summary")
    ._().Log(lambda b: f"Summary length: {len(b.summary)}")
    .End()
)

style_tree = (
    Tree[Blackboard]("StyleTree")
    .Sequence()
    ._().If(lambda b: b.use_formal)
    ._()._().Function(set_formal)
    ._()._().Else()
    ._()._()._().Function(set_casual)
    ._().WriteBlackboard("style")
    ._().Log(lambda b: f"Style: {b.style}")
    .End()
)

guess_tree = (
    Tree[Blackboard]("GuessTree")
    .Retry(3, sleep_secs=1)
    ._().Sequence()
    ._()._().LLM("openrouter/openai/gpt-5-nano", make_guess_messages)
    ._()._().ParseJSON(dst="guess_json")
    ._()._().Function(validate_guess)
    .End()
)

title_tree_a = (
    Tree[Blackboard]("TitleA")
    .Sequence()
    ._().LLM("openrouter/openai/gpt-5-nano", make_title_messages)
    ._().WriteBlackboard("title")
    .End()
)

title_tree_b = (
    Tree[Blackboard]("TitleB")
    .Sequence()
    ._().LLM("openrouter/google/gemma-3-27b-it:free", make_title_messages)
    ._().WriteBlackboard("title")
    .End()
)

title_tree_c = (
    Tree[Blackboard]("TitleC")
    .Sequence()
    ._().LLM("openrouter/meta-llama/llama-3.1-8b-instruct:free", make_title_messages)
    ._().WriteBlackboard("title")
    .End()
)

title_selector_tree = (
    Tree[Blackboard]("TitleSelector")
    .RandomSelector()
    ._().Subtree(title_tree_a)
    ._().Subtree(title_tree_b)
    ._().Subtree(title_tree_c)
    .End()
)

parallel_tree = (
    Tree[Blackboard]("ParallelTasks")
    .Parallel(concurrency_limit=3)
    ._().Subtree(summary_tree)
    ._().Subtree(style_tree)
    ._().Subtree(guess_tree)
    .End()
)

tree = (
    Tree[Blackboard]("ComplexTree")
    .Sequence()
    ._().Log("Start complex tree")
    ._().Function(init_problem)
    ._().Subtree(parallel_tree)
    ._().Subtree(title_selector_tree)
    ._().Log(lambda b: f"Final title: {b.title}")
    .End()
)
# fmt: on


async def main() -> None:
    context = Context()
    blackboard = Blackboard(prompt="A short story about a robot learning empathy.")

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("Result:", result)
    print("Summary:", blackboard.summary)
    print("Style:", blackboard.style)
    print("Title:", blackboard.title)

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:5173/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())

"""Retry decorator with an LLM call.

Retries the LLM until it guesses a prepared number 1-5 (max 5 tries).
"""

import asyncio
import os
import random
import sys
from dataclasses import dataclass

import litellm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable
from tinytasktree import JSON, Context, FileTraceStorageHandler, Result, Tree

litellm.suppress_debug_info = True
litellm.set_verbose = False

# Requirements:
#   - OPENROUTER_API_KEY set for OpenRouter access via LiteLLM


@dataclass
class Blackboard:
    expected: int = 0
    parsed: JSON | None = None
    last_guess: int | None = None
    hint: str = ""


def make_messages(b: Blackboard) -> list[JSON]:
    hint = ""
    if b.hint:
        last_guess = "unknown" if b.last_guess is None else str(b.last_guess)
        hint = f" Last guess: {last_guess}. Hint: {b.hint}."
    prompt = (
        "Guess the secret number (1-5)." + hint + " Return only valid JSON with no extra text. "
        'Format: {"guess": number}'
    )
    return [{"role": "user", "content": prompt}]


def on_delta(b: Blackboard, fulltext: str, delta: str, finished: bool) -> None:
    print(delta, end="", flush=True)


def init_problem(b: Blackboard) -> None:
    b.expected = random.randint(1, 5)


def validate_answer(b: Blackboard) -> Result:
    if not b.parsed:
        b.hint = "invalid JSON"
        return Result.FAIL(None)
    try:
        value = int(b.parsed["guess"])
    except (KeyError, TypeError, ValueError):
        b.hint = "invalid JSON"
        return Result.FAIL(None)
    b.last_guess = value
    if value == b.expected:
        b.hint = ""
        return Result.OK(b.parsed)
    b.hint = "too small" if value < b.expected else "too large"
    return Result.FAIL(b.parsed)


# fmt: off
tree = (
    Tree[Blackboard]("RetryLLM")
    .Sequence()
    ._().Function(init_problem)
    ._().Retry(5, sleep_secs=1)  # Up to 5 attempts
    ._()._().Sequence()
    ._()._()._().LLM("openrouter/openai/gpt-5-nano", make_messages, stream=True, stream_on_delta=on_delta)
    ._()._()._().ParseJSON(dst="parsed")
    ._()._()._().Function(validate_answer)
    .End()
)
# fmt: on


async def main() -> None:
    context = Context()
    blackboard = Blackboard()

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("\nExpected: [hidden]")
    print("\nResult:", result)
    print("Parsed:", blackboard.parsed)

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:5173/{trace_id}")

    # prevent litellm introducing wired at-exit warnings..

    await litellm.close_litellm_async_clients()


if __name__ == "__main__":
    asyncio.run(main())

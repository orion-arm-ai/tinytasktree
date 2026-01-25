"""Basic tree execution tests.

Steps:
- Configure the LLM mock with a known JSON response.
- Build a simple tree: LLM -> ParseJSON.
- Run the tree and assert parsing results.
Expectations:
- Tree returns OK status.
- Parsed JSON matches the mocked response.
"""

from __future__ import annotations

from dataclasses import dataclass

import tinytasktree


@dataclass
class Blackboard:
    prompt: str
    parsed: dict | None = None


def make_messages(b: Blackboard) -> list[tinytasktree.JSON]:
    return [{"role": "user", "content": b.prompt}]


async def test_simple_tree_llm_parse_json(mock_litellm):
    mock_litellm(content='{"answer": "ok", "count": 2}')
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("SimpleTree")
        .Sequence()
        ._().LLM("mock/model", make_messages)
        ._().ParseJSON(dst="parsed")
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="Return a JSON object")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert blackboard.parsed == {"answer": "ok", "count": 2}

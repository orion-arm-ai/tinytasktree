"""ParseJSON node behavior tests.

Steps:
- Parse valid JSON from last_result, from blackboard attr, and via custom getters/setters.
- Parse JSON wrapped in ```json fences.
- Confirm the default loader falls back to strict parsing when json_repair is unavailable.
- Confirm the default loader prefers json_repair when it is available.
- Confirm custom loaders also receive fence-stripped input.
Expectations:
- Valid JSON parses and returns OK with parsed data.
- JSON fences are stripped before parsing.
- Invalid JSON fails when json_repair is unavailable.
- Repairable JSON succeeds when json_repair is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import tinytasktree


@dataclass
class Blackboard:
    raw: str = ""
    parsed: dict | None = None
    captured: dict | None = None


def set_captured(b: Blackboard, data: dict) -> None:
    b.captured = data


def get_raw(b: Blackboard) -> str:
    return b.raw


async def test_parse_json_valid_from_last_result():
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("ParseFromLast")
        .Sequence()
        ._().Function(lambda: '{"a": 1}')
        ._().ParseJSON(dst="parsed")
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == {"a": 1}
    assert blackboard.parsed == {"a": 1}


async def test_parse_json_from_blackboard_attr_and_custom_dst():
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("ParseFromAttr")
        .Sequence()
        ._().ParseJSON(src="raw", dst=set_captured)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard(raw='{"b": 2}')
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == {"b": 2}
    assert blackboard.captured == {"b": 2}


async def test_parse_json_from_custom_src_and_dst_attr():
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("ParseCustomSrc")
        .Sequence()
        ._().ParseJSON(src=get_raw, dst="parsed")
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard(raw='{"c": 3}')
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == {"c": 3}
    assert blackboard.parsed == {"c": 3}


async def test_parse_json_with_fenced_json():
    fenced = """```json\n{\"d\": 4}\n```"""

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("ParseFenced")
        .Sequence()
        ._().Function(lambda: fenced)
        ._().ParseJSON(dst="parsed")
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == {"d": 4}
    assert blackboard.parsed == {"d": 4}


async def test_parse_json_bad_json_default_loader_fails():
    bad = '{"e": 5'
    original_json_repair = tinytasktree.json_repair
    tinytasktree.json_repair = None

    try:
        # fmt: off
        tree = (
            tinytasktree.Tree[Blackboard]("ParseStrictInvalid")
            .Sequence()
            ._().Function(lambda: bad)
            ._().ParseJSON(dst="parsed")
            .End()
        )
        # fmt: on

        context = tinytasktree.Context()
        blackboard = Blackboard()
        async with context.using_blackboard(blackboard):
            result = await tree(context)

        assert not result.is_ok()
        assert result.data == bad
        assert blackboard.parsed is None
    finally:
        tinytasktree.json_repair = original_json_repair


async def test_parse_json_default_loader_uses_json_repair_when_available():
    bad = '{"e": 5'

    class FakeJsonRepair:
        @staticmethod
        def loads(s: str) -> dict:
            assert s == bad
            return {"e": 5}

    original_json_repair = tinytasktree.json_repair
    tinytasktree.json_repair = FakeJsonRepair()

    try:
        # fmt: off
        tree = (
            tinytasktree.Tree[Blackboard]("ParseRepairableDefaultLoader")
            .Sequence()
            ._().Function(lambda: bad)
            ._().ParseJSON(dst="parsed")
            .End()
        )
        # fmt: on

        context = tinytasktree.Context()
        blackboard = Blackboard()
        async with context.using_blackboard(blackboard):
            result = await tree(context)

        assert result.is_ok()
        assert result.data == {"e": 5}
        assert blackboard.parsed == {"e": 5}
    finally:
        tinytasktree.json_repair = original_json_repair


async def test_parse_json_custom_loader_receives_stripped_text():
    captured = {}
    fenced = """```json\n{\"d\": 4}\n```"""

    def capture_loader(s: str) -> dict | None:
        captured["input"] = s
        return {"d": 4}

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("ParseCustomLoaderWithFences")
        .Sequence()
        ._().Function(lambda: fenced)
        ._().ParseJSON(dst="parsed", json_loader=capture_loader)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert captured["input"] == '{"d": 4}'
    assert blackboard.parsed == {"d": 4}


async def test_parse_json_bad_json_unrepairable():
    bad = "{ this is not json }"

    def always_fail(_: str) -> dict | None:
        return None

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("ParseUnrepairable")
        .Sequence()
        ._().Function(lambda: bad)
        ._().ParseJSON(dst="parsed", json_loader=always_fail)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert not result.is_ok()
    assert result.data == "{ this is not json }"
    assert blackboard.parsed is None

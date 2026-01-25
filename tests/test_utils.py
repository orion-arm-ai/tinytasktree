"""Helper function tests.

Steps:
- Verify deterministic output and permutation behavior for _weighted_shuffle.
- Check numeric conversion rules for _as_int.
- Validate string conversion for dict/list/dataclass in _try_to_string.
- Validate _orjson_default_serializer for supported types.
- Verify parameter counting including functools.partial in _inspect_func_parameters_count.
Expectations:
- Helpers return expected values for representative inputs.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum

import tinytasktree


class Color(Enum):
    RED = "red"


@dataclass
class Box:
    name: str
    tags: set[str]

    def json(self):
        return {"name": self.name, "tags": sorted(self.tags)}


class DictBox:
    def __init__(self, name: str):
        self.name = name

    def dict(self):
        return {"name": self.name}


def test_weighted_shuffle():
    items = ["a", "b", "c", "d"]

    random.seed(0)
    assert tinytasktree._weighted_shuffle(items) == ["d", "b", "a", "c"]

    random.seed(0)
    assert tinytasktree._weighted_shuffle(items, [1, 2, 3, 4]) == ["b", "a", "c", "d"]

    random.seed(1)
    shuffled = tinytasktree._weighted_shuffle(items)
    assert sorted(shuffled) == sorted(items)


def test_as_int():
    assert tinytasktree._as_int(None) is None
    assert tinytasktree._as_int("3") == 3
    assert tinytasktree._as_int(3.7) == 3
    assert tinytasktree._as_int("bad") is None


def test_try_to_string():
    assert tinytasktree._try_to_string({"a": 1}) == '{"a":1}'
    assert tinytasktree._try_to_string([1, "x"]) == "[1,x]"
    assert tinytasktree._try_to_string(Box(name="box", tags={"b", "a"})) in {
        '{"name":"box","tags":["a","b"]}',
        '{"name":"box","tags":["b","a"]}',
    }


def test_orjson_default_serializer():
    assert tinytasktree._orjson_default_serializer({"a", "b"}) in (["a", "b"], ["b", "a"])
    assert tinytasktree._orjson_default_serializer(date(2024, 1, 2)) == "2024-01-02"
    assert tinytasktree._orjson_default_serializer(datetime(2024, 1, 2, 3, 4, 5)) == "2024-01-02 03:04:05"
    assert tinytasktree._orjson_default_serializer(timedelta(seconds=2.5)) == 2.5
    assert tinytasktree._orjson_default_serializer(Color.RED) == "red"
    assert tinytasktree._orjson_default_serializer(Box(name="box", tags={"z"})) == {
        "name": "box",
        "tags": ["z"],
    }
    assert tinytasktree._orjson_default_serializer(DictBox(name="d")) == {"name": "d"}


def test_inspect_func_parameters_count():
    def f(a, b):
        return a + b

    def g(a, b, c):
        return a + b + c

    assert tinytasktree._inspect_func_parameters_count(f) == 2
    assert tinytasktree._inspect_func_parameters_count(lambda: None) == 0

    partial = functools.partial(g, 1)
    assert tinytasktree._inspect_func_parameters_count(partial) == 2

from __future__ import annotations

import tinytasktree


async def test_log_node_default_fullname_includes_level():
    tree = tinytasktree.Tree("LogTree").Log("hello").End()

    child = tree.child()
    assert child.fullname == "Log(info)"

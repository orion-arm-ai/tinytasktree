"""Builder customization tests.

Steps:
- Define a custom DecoratorNode and a Tree subclass that exposes it via _attach.
- Build a tree using the custom builder method.
Expectations:
- Custom node executes and returns its forced result.
"""

from __future__ import annotations

from dataclasses import dataclass

import tinytasktree


@dataclass
class Blackboard:
    value: str = ""


class ForceValueNode(tinytasktree.SingleChildNode[Blackboard], tinytasktree.DecoratorNode[Blackboard]):
    KIND = "ForceValue"

    def __init__(self, value: str, name: str = "") -> None:
        tinytasktree.DecoratorNode.__init__(self, name)
        tinytasktree.SingleChildNode.__init__(self, None, name)
        self._value = value

    async def _impl(self, context: tinytasktree.Context, tracer: tinytasktree.Tracer) -> tinytasktree.Result:
        child = self.child()
        async with context._forward(child.fullname):
            await child(context)
        return tinytasktree.Result.OK(self._value)


class CustomTree(tinytasktree.Tree[Blackboard]):
    def ForceValue(self, value: str, name: str = "") -> "CustomTree":
        return self._attach(ForceValueNode(value, name))


class MarkerLeafNode(tinytasktree.LeafNode[Blackboard]):
    KIND = "MarkerLeaf"

    def __init__(self, marker: list[str], name: str = "") -> None:
        super().__init__(name)
        self._marker = marker

    def OnBuildEnd(self) -> None:
        super().OnBuildEnd()
        self._marker.append("built")

    async def _impl(self, context: tinytasktree.Context, tracer: tinytasktree.Tracer) -> tinytasktree.Result:
        return tinytasktree.Result.OK("ok")


class CustomLeafTree(tinytasktree.Tree[Blackboard]):
    def MarkerLeaf(self, marker: list[str], name: str = "") -> "CustomLeafTree":
        return self._attach(MarkerLeafNode(marker, name))


async def test_custom_builder_method():
    def write_child_value(b: Blackboard) -> str:
        b.value = "child"
        return b.value

    # fmt: off
    tree = (
        CustomTree("CustomTree")
        .ForceValue("ok")
        ._().Function(write_child_value)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "ok"
    assert blackboard.value == "child"


async def test_attach_leaf_node_calls_on_build_end():
    marker: list[str] = []

    tree = CustomLeafTree("CustomLeafTree").MarkerLeaf(marker).End()

    assert marker == ["built"]
    assert tree.child().fullname == "MarkerLeaf"

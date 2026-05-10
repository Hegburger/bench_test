"""IDS (Ideographic Description Sequences) parser.

Parses IDS strings like "⿰至⿱⿰先先貝" into a tree structure where:
- Internal nodes are IDS operators (⿰ ⿱ ⿴ ⿻ etc., U+2FF0-U+2FFF)
- Leaf nodes are regular characters (CJK, components, etc.)

The tree can then be used for tree edit distance computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IDSNode:
    """A node in an IDS parse tree."""

    value: str  # The character/operator at this node
    children: list[IDSNode] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_operator(self) -> bool:
        return len(self.children) > 0

    def preorder(self) -> list[str]:
        """Return preorder traversal labels."""
        result = [self.value]
        for child in self.children:
            result.extend(child.preorder())
        return result

    def postorder(self) -> list[str]:
        """Return postorder traversal labels for tree edit distance."""
        result = []
        for child in self.children:
            result.extend(child.postorder())
        result.append(self.value)
        return result

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "children": [c.to_dict() for c in self.children],
        }

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"IDSNode({self.value!r})"
        children_repr = ", ".join(repr(c) for c in self.children)
        return f"IDSNode({self.value!r}, [{children_repr}])"


# IDS operators and their arities
IDS_OPERATORS = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")

# Binary operators (left-right, top-bottom, surround, overlay, etc.)
IDS_BINARY = set("⿰⿱⿴⿵⿶⿷⿸⿹⿺⿻")

# Trinary operators (left-middle-right, top-middle-bottom)
IDS_TRINARY = set("⿲⿳")


def parse_ids(ids_str: str) -> IDSNode | None:
    """Parse an IDS string into a tree.

    Returns None if the string is not a valid IDS (no operator found).
    For empty strings, returns None.
    For a single non-operator character, returns a leaf node.
    """
    ids_str = ids_str.strip()
    if not ids_str:
        return None

    try:
        node, pos = _parse(ids_str, 0)
        return node
    except (IndexError, ValueError):
        return None


def _parse(s: str, pos: int) -> tuple[IDSNode, int]:
    """Recursive descent parser. Returns (node, next_pos)."""
    if pos >= len(s):
        raise ValueError(f"Unexpected end of string at position {pos}")

    ch = s[pos]
    if ch in IDS_OPERATORS:
        node = IDSNode(value=ch)
        if ch in IDS_TRINARY:
            arity = 3
        else:
            arity = 2

        pos += 1  # consume operator
        for _ in range(arity):
            child, pos = _parse(s, pos)
            node.children.append(child)
        return node, pos
    else:
        # Leaf: consume one character (which may be a CJK char or multi-codepoint)
        # For simplicity, consume one Python character.
        # CJK unified ideographs are single codepoints in Python strings.
        return IDSNode(value=ch), pos + 1


def ids_string_normalized(ids_str: str) -> str:
    """Normalize an IDS string to standard form.

    Handles common variants:
    - Strips whitespace
    - Unicode NFKC normalization
    """
    import unicodedata
    return unicodedata.normalize("NFKC", ids_str.strip())

"""Utilities for exporting trees in nice forms."""


def to_syntax_tree(tree: list) -> str:
    if isinstance(tree, str):
        return "[" + tree.replace("[", "{").replace("]", "}") + "]"
    else:
        child_strings = [to_syntax_tree(child) for child in tree]
        return "[S " + " ".join(child_strings) + "]"


def to_nested_indents(tree: list, depth: int = 0) -> str:
    if not isinstance(tree, list):
        return depth * "    " + repr(tree)
    else:
        return "\n".join(to_nested_indents(child, depth + 1) for child in tree)

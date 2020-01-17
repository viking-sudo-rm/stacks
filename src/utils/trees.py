from typing import List, Iterator
import json
from nltk.tree import Tree


# See https://www.clips.uantwerpen.be/pages/mbsp-tags.
_PUNCTUATION_TAGS = {".", ",", ":", "(", ")"}


def _gen_flat_tree(tree) -> Iterator[str]:
    """Return all the tokens in the tree."""
    if not isinstance(tree, list):
        yield tree
    else:
        for child in tree:
            yield from _gen_flat_tree(child)


def flatten_tree(tree) -> List[str]:
    return list(_gen_flat_tree(tree))


def right_distances(tree, distance=0) -> List[int]:
    if not isinstance(tree, list):
        return [distance]
    elif len(tree) == 2:
        left_seq = right_distances(tree[0], distance + 1)
        right_seq = right_distances(tree[1], 0)
        return left_seq + right_seq
    else:
        raise ValueError("Non-binary node in tree")


def left_distances(tree, distance=0) -> List[int]:
    if not isinstance(tree, list):
        return [distance]
    elif len(tree) == 2:
        left_seq = left_distances(tree[0], 0)
        right_seq = left_distances(tree[1], distance + 1)
        return left_seq + right_seq
    else:
        raise ValueError("Non-binary node in tree")


def brackets(tree) -> List[str]:
    # Unlabelled 1-Dyck representation.
    if not isinstance(tree, list):
        return []
    elif len(tree) == 2:
        seq = ["("]
        seq.extend(brackets(tree[0]))
        seq.extend(brackets(tree[1]))
        seq.append(")")
        return seq
    else:
        raise ValueError("Non-binary node in tree")


def from_left_distances(sentence, left_dists, return_stack: bool = False):
    """Construct a parse tree from a left_dists vector.
    Return None if the vectorization is invalid."""
    stack = []
    for const, dist in zip(sentence, left_dists):
        for _ in range(dist):
            if not stack:
                return None
            const = [stack.pop(), const]
        stack.append(const)

    if return_stack:
        return stack[-1], stack
    else:
        return stack[-1]


def from_right_distances(sentence, right_dists):
    tree = from_left_distances(reversed(sentence), reversed(right_dists))
    return reverse_tree(tree)


def reverse_tree(tree):
    if not isinstance(tree, list):
        return tree
    else:
        return [reverse_tree(node) for node in reversed(tree)]


def binarize(tree):
    """Binarize a tree and remove null PTB nodes."""
    if not isinstance(tree, list):
        if tree.startswith("*"):
            return None
        else:
            return tree
    else:
        const = None
        for node in reversed(tree):
            left = binarize(node)
            if left is not None:
                const = left if const is None else [left, const]
        return const


def get_pos_tags(tree: Tree, bert_tags: bool = False, eos: bool = False) -> List[str]:
    tags = list(_gen_pos_tags(tree))
    if bert_tags:
        tags.insert(0, "#")
        tags.append("#")
    if eos:
        tags.append("#")
    return tags


def _gen_pos_tags(tree: Tree) -> List[str]:
    """Return the POS tags from an NLTK tree."""
    if len(tree) == 1 and not isinstance(tree[0], Tree):
        if not tree[0].startswith("*"):
            yield tree.label()
    else:
        for child in tree:
            yield from _gen_pos_tags(child)


def remove_punctuation(tree: Tree) -> list:
    if len(tree) == 1 and not isinstance(tree[0], Tree):
        if tree.label() in _PUNCTUATION_TAGS:
            return None
        else:
            return tree
    else:
        children = [remove_punctuation(child) for child in tree]
        children = [child for child in children if child is not None]
        return Tree(tree.label(), children)


def get_right_branching(tokens: List[str]) -> list:
    const = None
    for token in reversed(tokens):
        const = token if const is None else [token, const]
    return const


def add_bert_tags(tree: list) -> list:
    return ["[CLS]", [tree, "[SEP]"]]


def add_eos(tree: list) -> list:
    return [tree, "<eos>"]

def to_syntax_tree(tree: list):
    if isinstance(tree, str):
        return "[" + tree.replace("[", "{").replace("]", "}") + "]"
    else:
        child_strings = [to_syntax_tree(child) for child in tree]
        return "[S " + " ".join(child_strings) + "]"

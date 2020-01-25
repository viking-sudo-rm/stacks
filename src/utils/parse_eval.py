from typing import List
import numpy as np

def get_brackets(tree, idx=0):
    """Taken from
    https://github.com/yikangshen/PRPN/blob/master/test_phrase_grammar.py"""
    # TODO: Compare to to_indexed_contituents in Htut et al.
    brackets = set()
    if isinstance(tree, list):
        for node in tree:
            node_brac, next_idx = get_brackets(node, idx)
            if next_idx - idx > 1:
                brackets.add((idx, next_idx))
                brackets.update(node_brac)
            idx = next_idx
        return brackets, idx
    else:
        return brackets, idx + 1


def _get_f1(gold_tree, our_tree):
    """Taken from
    https://github.com/yikangshen/PRPN/blob/master/test_phrase_grammar.py"""
    model_out, _ = get_brackets(our_tree)
    std_out, _ = get_brackets(gold_tree)
    overlap = model_out.intersection(std_out)

    # For unlabelled binary trees, precision = recall = F1.
    precision = float(len(overlap)) / (len(model_out) + 1e-8)
    return precision


def get_batched_f1(gold_trees: List[list], pred_trees: List[list]) -> int:
    # Require lists and this check, since zipping two generators together will not throw an error if
    # the lengths do not match.
    assert len(gold_trees) == len(pred_trees), "Mismatched number of trees."
    f1s = [_get_f1(gold, pred) for gold, pred in zip(gold_trees, pred_trees)]
    return np.mean(f1s)

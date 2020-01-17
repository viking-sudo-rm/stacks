from typing import List
import numpy as np
import random
import sys
from listify import listify

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer

from src.utils.trees import from_right_distances


MODES = ["bool", "int"]
ARGS = [["F", "T"], ["0", "1"]]
OPS = ["*", "+"]


def _parse_path(path: str) -> List[int]:
    pieces = [int(x) for x in path.split(":")]
    if len(pieces) == 3:
        return pieces
    elif len(pieces) == 2:
        return pieces[0], pieces[1], 2
    else:
        raise ValueError("Unknown path specification: " + path)


@DatasetReader.register("eval")
class EvalReader(DatasetReader):

    modes = ["bool", "int"]

    def __init__(self,
                 p_unary: float = .25,
                 p_binary: float = .5,
                 add_lengths: bool = False):
        super().__init__(lazy=True)
        self.p_unary = p_unary
        self.p_binary = p_binary
        self.add_lengths = add_lengths
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}

    def _read(self, path: str):
        num_samples, length, seed = _parse_path(path)
        random.seed(seed)
        for _ in range(num_samples):
            tokens = self._sample_tokens(length)
            yield self.text_to_instance(tokens)

    def text_to_instance(self, tokens):
        fields = {
            "source": TextField([Token(tok) for tok in tokens], self.token_indexers),
        }

        if self.add_lengths:
            length = len(fields["source"])
            fields["lengths"] = LabelField(length, skip_indexing=True)

        return Instance(fields)

    def _sample_tokens(self, length: int, mode: int = 0):
        if length <= 0:
            choices = ARGS[mode]
            return [random.choice(choices)]

        oracle = random.random()
        if oracle < self.p_unary:
            tokens = self._sample_tokens(length - 1, 1 - mode)
            tokens.insert(0, MODES[mode])
            return tokens

        elif oracle < self.p_unary + self.p_binary:
            tokens = [OPS[mode]]
            tokens.extend(self._sample_tokens(length - 1, mode))
            tokens.extend(self._sample_tokens(length - len(tokens), mode))
            return tokens

        else:
            choices = ARGS[mode]
            return [random.choice(choices)]

    @listify
    def get_binary_trees(self, path: str) -> List[list]:
        """Return binary trees sampled from distribution specified by a data path."""
        for instance in self.read(path):
            tokens = [tok.text for tok in instance["source"]]
            yield self._get_binary_tree(tokens)

    @staticmethod
    def _get_binary_tree(tokens: List[str]) -> list:
        actions = []
        for token in tokens:
            if token in OPS:
                actions.append(2)
            elif token in MODES:
                actions.append(1)
            else:
                actions.append(0)

        return from_right_distances(tokens, actions)

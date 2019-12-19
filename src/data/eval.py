from typing import List
import numpy as np
import random
import sys

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer


MODES = ["bool", "int"]
ARGS = ["FT", "01"]
OPS = ["*", "+"]


@DatasetReader.register("eval")
class EvalReader(DatasetReader):

    modes = ["bool", "int"]

    def __init__(self, p_unary: float = .25, p_binary: float = .5):
        super().__init__(lazy=True)
        self.p_unary = p_unary
        self.p_binary = p_binary
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}

    def _read(self, path: str):
        num_samples, length = [int(x) for x in path.split(":")]
        for _ in range(num_samples):
            tokens = self._sample_tokens(length)
            yield self.text_to_instance(tokens)

    def text_to_instance(self, tokens):
        # Language modeling version. TODO: Also implement weird multiclass version.
        tokens = TextField([Token(tok) for tok in tokens], self.token_indexers)
        fields = {"source": tokens}
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

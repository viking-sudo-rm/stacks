from typing import List
import numpy as np
import random

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer


@DatasetReader.register("predict_opening")
class PredictOpeningGenerator(DatasetReader):

    """This is very similar to PushPop, except that the distribution of open/close actions leads to
    deeper embedding on the stack.
    Phrases are always expanded to depth.. although nonterminal rules are chosen randomly between
    center and left embedding.
    """

    def __init__(self,
                 op_range: int = 2,
                 center_embed_prob: float = .8):
        super().__init__(lazy=False)
        self.tokens = []
        open_tokens = ["o" + str(op) for op in range(op_range)]
        close_tokens = ["c" + str(op) for op in range(op_range)]
        self.token_pairs = list(zip(open_tokens, close_tokens))
        self.center_embed_prob = center_embed_prob
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}

    def _read(self, path: str):
        num_samples, max_depth = [int(x) for x in path.split(":")]
        for idx in range(num_samples):
            tokens = self._sample_tokens(max_depth)
            tags = self._get_tags(tokens)
            yield self.text_to_instance(tokens, tags)

    def text_to_instance(self, tokens, tags=None):
        tokens = TextField([Token(tok) for tok in tokens], self.token_indexers)
        fields = {"tokens": tokens}

        if tags is not None:
            fields.update(tags=SequenceLabelField(tags, sequence_field=tokens))

        return Instance(fields)

    def _sample_tokens(self, max_depth: int, depth: int = 0):
        if depth == max_depth:
            # Terminate the phrase at max depth.
            return []
        elif random.random() < self.center_embed_prob:
            # Expand via center embedding.
            tokens = self._sample_tokens(max_depth, depth + 1)
            o, c = random.choice(self.token_pairs)
            tokens.insert(0, o)
            tokens.append(c)
            return tokens
        else:
            # Expand via left embedding.
            tokens = self._sample_tokens(max_depth, depth + 1)
            tokens.extend(self._sample_tokens(max_depth, depth + 1))
            return tokens

    def _get_tags(self, tokens: List[str]):
        tags = []
        stack = []
        for token in tokens:
            if token.startswith("o"):
                stack.append(token)
            else:
                stack.pop()
            tags.append(stack[0] if stack else "0")
        return tags

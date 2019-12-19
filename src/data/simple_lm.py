from typing import Dict
from overrides import overrides

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


@DatasetReader.register("simple_lm")
class SimpleLmReader(DatasetReader):

    """A basic reader for these LM datasets."""

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 character_level: bool = False,
                 start_token: str = None,
                 end_token: str = None,
                 min_length: int = 2,
                 two_fields: bool = False):
        super().__init__(lazy=False)
        self._character_level = character_level
        self._start_token = start_token
        self._end_token = end_token
        self._min_length = min_length
        self._two_fields = two_fields
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("="):
                    continue

                tokens = list(line) if self._character_level else line.split(" ")
                if self._start_token is not None:
                    tokens.insert(0, self._start_token)
                if self._end_token is not None:
                    tokens.append(self._end_token)

                if len(tokens) < self._min_length:
                    return None
                yield self.text_to_instance(tokens)

    @overrides
    def text_to_instance(self, tokens):
        if not self._two_fields:
            # Return a single "source" field. This is the format of the standard AllenNLP LM.
            source = TextField([Token(t) for t in tokens], self._token_indexers)
            return Instance({"source": source})

        else:
            # Return a "source" field and a "target" field of labels.
            source = TextField([Token(t) for t in tokens[:-1]],
                               self._token_indexers)
            target = TextField([Token(t) for t in tokens[1:]],
                               self._token_indexers)
            return Instance({
                "source": source,
                "target": target,
            })

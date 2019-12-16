from typing import List

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer


@DatasetReader.register("tag-seq-reader")
class TaggedSequenceReader(DatasetReader):

    """Read data in the following format: tokn/tagn tokn/tagn .. tokn/tagn"""

    def __init__(self):
        super().__init__(lazy=True)
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}

    def _read(self, path: str):
        with open(path) as in_file:
            for line in in_file:
                if not line:
                    continue
                line = line.strip().split(" ")
                tokens, tags = zip(*[pair.split("/") for pair in line])
                yield self.text_to_instance(tokens, tags)

    def text_to_instance(self, tokens, tags=None):
        tokens = TextField([Token(tok) for tok in tokens], self.token_indexers)
        fields = {"tokens": tokens}

        if tags is not None:
            fields.update(tags=SequenceLabelField(tags, sequence_field=tokens))

        return Instance(fields)

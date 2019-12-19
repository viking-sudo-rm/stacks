from typing import List
import numpy as np
import random
import sys

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer


@DatasetReader.register("dyck")
class DyckReader(DatasetReader):

    """Reader used to generate a dataset for the arentheses task defined by Suzgun, et al. (2019):
    https://arxiv.org/pdf/1911.03329.pdf

    Reads data generated by the following class:
    https://github.com/suzgunmirac/marnns/blob/master/tasks/dyck_generator.py

    To load the data, use a TaggedSequenceReader.
    """

    def __init__(self):
        super().__init__(lazy=True)
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}

    def _read(self, path: str):
        with open(path) as in_file:
            for line in in_file:
                if not line:
                    continue

                tokens = [tok for tok in line.strip()]
                yield self.text_to_instance(tokens)

    def text_to_instance(self, tokens):
        # Language modeling version. TODO: Also implement weird multiclass version.
        tokens = TextField([Token(tok) for tok in tokens], self.token_indexers)
        fields = {"source": tokens}
        return Instance(fields)

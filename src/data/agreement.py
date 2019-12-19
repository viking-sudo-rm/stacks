from typing import Dict, List, Optional

from overrides import overrides

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer


@DatasetReader.register("agreement")
class AgreementReader(DatasetReader):

    def __init__(self,
                 token_indexers: Optional[Dict[str, TokenIndexer]],
                 lazy: bool = True):
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, path: str):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue

                label, tokens = line.split("\t")
                tokens = tokens.split()
                yield self.text_to_instance(tokens, label)

    @overrides
    def text_to_instance(self, tokens: List[str], label: str):
        tokens_field = TextField([Token(token) for token in tokens], self._token_indexers)
        label_field = LabelField(label)
        
        return Instance({
            "tokens": tokens_field,
            "label": label_field,
        })

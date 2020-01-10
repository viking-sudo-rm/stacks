from typing import Dict, Tuple, List
import logging

from overrides import overrides
from conllu import parse_incr

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("ud-arc-standard")
class UdArcStandardReader(DatasetReader):

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)
        with open(file_path, "r") as conllu_file:
            logger.info("Reading arc-standard actions from conllu dataset at: %s", file_path)
            for annotation in parse_incr(conllu_file):
                annotation = [x for x in annotation if isinstance(x["id"], int)]
                words = [x["form"] for x in annotation]
                heads = [x["head"] for x in annotation]
                actions = self.get_actions(heads)
                assert len(actions) == 2 * len(words) - 1
                yield self.text_to_instance(words, actions)

    @overrides
    def text_to_instance(self, words: List[str], actions: List[str]) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = [Token(t) for t in words]
        fields["tokens"] = TextField(tokens, self._token_indexers)
        # Construct a SequenceLabelField corresponding to 2n - 1 positions.
        pos_field = TextField([Token(str(x)) for x in range(2 * len(words) - 1)],
                              self._token_indexers)
        # We use "actions" as a namespace so that padding is added/easily check padding in model.
        fields["actions"] = SequenceLabelField(actions, pos_field, label_namespace="actions")
        return Instance(fields)

    def get_actions(self, heads):
        heads = [head - 1 for head in heads]
        dependents = [[] for head in heads]
        for idx, head in enumerate(heads):
            if head >= 0:
                dependents[head].append(idx)

        actions = []
        for idx in range(len(heads)):
            actions.append("shift")
            for dep in dependents[idx]:
                if dep < idx:
                    actions.append("left")
            head = heads[idx]
            if 0 <= head < idx:
                actions.append("right")

        return actions

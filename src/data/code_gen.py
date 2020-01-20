from typing import Dict, Optional, List
from overrides import overrides
import random

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


@DatasetReader.register("code_gen")
class CodeGenReader(DatasetReader):

    """A reader for character-level code generation.

    We split the code into blocks heuristically by splitting at \n\n."""

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 block_sep: str = "\n\n",
                 min_length: int = 5,
                 max_length: Optional[int] = None,
                 add_lengths: bool = False):
        super().__init__(lazy=False)
        self.block_sep = block_sep
        self.min_length = min_length
        self.max_length = max_length
        self.add_lengths = add_lengths
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        with open(file_path) as f:
            text = f.read()
            for block in text.split(self.block_sep):
                block = block.strip()
                if len(block) < self.min_length:
                    continue
                if self.max_length is not None:
                    block = block[:self.max_length]

                yield self.text_to_instance(block)

    @overrides
    def text_to_instance(self, text: str):
        fields = {
            "source": TextField([Token(t) for t in text], self.token_indexers)
        }
        
        if self.add_lengths:
            length = len(fields["source"])
            fields["lengths"] = LabelField(length, skip_indexing=True)

        return Instance(fields)


"""Methods for splitting code with train/test/dev split."""


def save_blocks(blocks: List[str], path: str) -> None:
    text = "\n\n".join(blocks)
    with open(path, "w") as f:
        f.write(text)


def split_train_valid_test(path: str, block_sep: str = "\n\n") -> None:
    with open(path) as f:
        text = f.read()
        blocks = [block.strip() for block in text.split(block_sep)]

    random.shuffle(blocks)
    end_train_idx = int(.8 * len(blocks))
    end_valid_idx = int(.9 * len(blocks))

    train_blocks = blocks[:end_train_idx]
    valid_blocks = blocks[end_train_idx:end_valid_idx]
    test_blocks = blocks[end_valid_idx:]
    return train_blocks, valid_blocks, test_blocks

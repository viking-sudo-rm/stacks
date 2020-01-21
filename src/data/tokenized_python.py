from typing import Dict, Optional, List
from tokenize import tokenize, COMMENT, NL, ENCODING
from token import NAME, STRING, NUMBER, INDENT, DEDENT
import os
from listify import listify
from overrides import overrides

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


_ALLOWED_NAMES = {
    "pass", "assert", "return", "print", "def", "class", "continue", "yield", "input", "len", "sum",
    "if", "elif", "else", "try", "except", "break",
    "any", "all", "self", "cls", "isinstance", "issubclass",
    "iter", "list", "str", "int", "float", "bool", "object", "buffer", "dict", "callable",
    "staticmethod", "classmethod", "abstractmethod", "overrides", "listify", "deprecated"
    "np", "torch", "tensor"
}


@DatasetReader.register("python")
class TokenizedPythonReader(DatasetReader):

    """A reader for word-tokenized python. Strings, comments, numbers, and symbolic names are
    stripped.
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 strip_names: bool = True,
                 strip_strings: bool = True,
                 strip_comments: bool = True,
                 strip_numbers: bool = True,
                 strip_indents: bool = True,  # Important because number of spaces depends on depth.
                 strip_dedents: bool = True,
                 strip_encodings: bool = True,
                 strip_aesthetic_newlines: bool = True,
                 max_length: Optional[int] = None,
                 add_lengths: bool = False):
        super().__init__(lazy=False)
        self.strip_names = strip_names
        self.strip_strings = strip_strings
        self.strip_comments = strip_comments
        self.strip_numbers = strip_numbers
        self.strip_indents = strip_indents
        self.strip_dedents = strip_dedents
        self.strip_encodings = strip_encodings
        self.strip_aesthetic_newlines = strip_aesthetic_newlines
        self.add_lengths = add_lengths
        self.max_length = max_length
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, dir_path: str):
        for file_name in os.listdir(dir_path):
            if file_name.endswith(".py"):
                path = os.path.join(dir_path, file_name)
                tokens = self._get_tokens(path)
                if self.max_length is not None:
                    tokens = tokens[:self.max_length]
                yield self.text_to_instance(tokens)

    @overrides
    def text_to_instance(self, tokens: str):
        fields = {
            "source": TextField([Token(t) for t in tokens], self.token_indexers)
        }
        
        if self.add_lengths:
            length = len(fields["source"])
            fields["lengths"] = LabelField(length, skip_indexing=True)

        return Instance(fields)

    @listify
    def _get_tokens(self, path: str) -> List[str]:
        with open(path, "rb") as fh:
            for tok_type, tok_string, _, _, _ in tokenize(fh.readline):
                if tok_type == STRING and self.strip_strings:
                    yield "<STRING>"
                elif tok_type == NUMBER and self.strip_numbers:
                    yield "<NUMBER>"
                elif tok_type == COMMENT and self.strip_comments:
                    yield "<COMMENT>"
                elif tok_type == NAME and self.strip_names and tok_string not in _ALLOWED_NAMES:
                    yield "<NAME>"
                elif tok_type == INDENT and self.strip_indents:
                    yield "<INDENT>"
                elif tok_type == DEDENT and self.strip_dedents:
                    yield "<DEDENT>"
                elif tok_type == ENCODING and self.strip_encodings:
                    continue
                elif tok_type == NL and self.strip_aesthetic_newlines:
                    continue
                else:
                    yield tok_string

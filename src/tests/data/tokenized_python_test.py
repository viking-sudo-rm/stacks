import os
import pathlib
from typing import cast

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.fields import TextField

from src.data.tokenized_python import TokenizedPythonReader


_TOKENS = ["hello", "world", "vossi", "vop", "."]


class TestSimpleLmReader(AllenNlpTestCase):
    
    FIXTURES = pathlib.Path(__file__).parent / ".." / "fixtures"
    PATH = FIXTURES

    def test_read(self):
        reader = TokenizedPythonReader()
        tokens = [[t.text for t in inst["source"]] for inst in reader.read(self.PATH)]
        exp_tokens = [['def', '<NAME>', '(', 'self', ')', ':', '\n', '<INDENT>', 'self', '.',
                       '<NAME>', '(', ')', '\n', '<DEDENT>', '']]
        assert tokens == exp_tokens

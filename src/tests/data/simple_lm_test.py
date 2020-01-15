import os
import pathlib
from typing import cast

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.fields import TextField

from src.data.simple_lm import SimpleLmReader


class TestSimpleLmReader(AllenNlpTestCase):
    
    FIXTURES = pathlib.Path(__file__).parent / ".." / "fixtures"
    PATH = os.path.join(FIXTURES, "lm.txt")

    def test_text_to_instance(self):
        reader = SimpleLmReader()
        sentence = ["The", "only", "sentence", "."]
        instance = reader.text_to_instance(sentence)
        text = [t.text for t in cast(TextField, instance.fields["source"]).tokens]
        self.assertEqual(text, sentence)

    def test_text_to_instance_two_fields(self):
        reader = SimpleLmReader(two_fields=True)
        sentence = ["The", "only", "sentence", "."]
        instance = reader.text_to_instance(sentence)

        expected_source = ["The", "only", "sentence"]
        source = [t.text for t in cast(TextField, instance.fields["source"]).tokens]
        self.assertEqual(source, expected_source)

        expected_target = ["only", "sentence", "."]
        target = [t.text for t in cast(TextField, instance.fields["target"]).tokens]
        self.assertEqual(target, expected_target)

    def test_read(self):
        reader = SimpleLmReader()
        tags = [[t.text for t in inst["source"]] for inst in reader.read(self.PATH)]
        expected_tags = [
            ["this", "is", "a", "sentence", "though", "!"],
            ["so", "is", "this", "."],
        ]
        assert tags == expected_tags

    def test_read_start_end(self):
        reader = SimpleLmReader(start_token="<s>", end_token="</s>")
        tags = [[t.text for t in inst["source"]] for inst in reader.read(self.PATH)]
        expected_tags = [
            ["<s>", "this", "is", "a", "sentence", "though", "!", "</s>"],
            ["<s>", "so", "is", "this", ".", "</s>"],
        ]
        assert tags == expected_tags

    def test_read_add_lengths(self):
        reader = SimpleLmReader(add_lengths=True)
        lengths = [inst["lengths"].label for inst in reader.read(self.PATH)]
        expected_lengths = [6, 4]
        assert lengths == expected_lengths

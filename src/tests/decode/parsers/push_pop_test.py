from allennlp.common.testing import AllenNlpTestCase

from src.decode.parsers import PushPopParser


class PushPopTest(AllenNlpTestCase):

    parser = PushPopParser()

    def test_get_parses(self):
        tokens = [x for x in "()(())"]
        actions = [0, 1, 0, 0, 1, 1]
        parse, stack = self.parser.get_parse(tokens, actions)

        exp_parse = [["(", ")"], ["(", ["(", ")"], ")"]]
        assert parse == exp_parse
        assert stack == [parse]

    def test_get_parses_inc(self):
        tokens = [x for x in "((())"]
        actions = [0, 0, 0, 1, 1]
        parse, stack = self.parser.get_parse(tokens, actions)

        exp_stack = [[], ["(", ["(", ["(", ")"], ")"]]]
        exp_parse = ["(", ["(", ["(", ")"], ")"]]
        assert parse == exp_parse
        assert stack == exp_stack

from allennlp.common.testing import AllenNlpTestCase

from src.decode.parsers import NoOpParser


class NoOpTest(AllenNlpTestCase):

    parser = NoOpParser()

    def test_get_parses(self):
        tokens = [x for x in "(hi)((yo))"]
        actions = [0, 1, 1, 2, 0, 0, 1, 1, 2, 2]
        parse, stack = self.parser.get_parse(tokens, actions)

        exp_parse = [["(", "h", "i", ")"], ["(", ["(", "y", "o", ")"], ")"]]
        assert parse == exp_parse
        assert stack == [parse]

    def test_get_parses_inc(self):
        tokens = [x for x in "(a(())b"]
        actions = [0, 1, 0, 0, 2, 2, 1]
        parse, stack = self.parser.get_parse(tokens, actions)

        exp_parse = ["(", "a", ["(", ["(", ")"], ")"], "b"]
        exp_stack = [[], exp_parse]
        assert parse == exp_parse
        assert stack == exp_stack

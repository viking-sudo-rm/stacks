from overrides import overrides
from typing import List, Tuple

from src.decode.parsers.base import ActionParser
from src.utils.minimalist import from_merges


@ActionParser.register("merge")
class MergeParser(ActionParser):

    @staticmethod
    @overrides
    def get_parse(tokens: List[str], actions: List[int]) -> Tuple[list, list]:
        return from_merges(tokens, actions)

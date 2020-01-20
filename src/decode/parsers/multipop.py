from overrides import overrides
from typing import List, Tuple

from src.decode.parsers.base import ActionParser
from src.utils.trees import from_left_distances


@ActionParser.register("kpop")
class MultipopParser(ActionParser):

    @staticmethod
    @overrides
    def get_parse(tokens: List[str], actions: List[int]) -> Tuple[list, list]:
        return from_left_distances(tokens, actions, return_stack=True)

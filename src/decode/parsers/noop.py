from overrides import overrides
from typing import List, Tuple

from src.decode.parsers.base import ActionParser


@ActionParser.register("noop")
class NoOpParser(ActionParser):

    @staticmethod
    @overrides
    def get_parse(tokens: List[str], actions: List[int]) -> Tuple[list, list]:
        stack = [[]]
        for token, action in zip(tokens, actions):
            if action == 0:
                stack.append([])

            stack[-1].append(token)

            if action == 2:
                const = stack.pop()
                stack[-1].append(const)

        return stack[-1], stack

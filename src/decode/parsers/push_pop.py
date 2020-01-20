from overrides import overrides
from typing import List, Tuple

from src.decode.parsers.base import ActionParser


@ActionParser.register("push-pop")
class PushPopParser(ActionParser):

    @staticmethod
    @overrides
    def get_parse(tokens: List[str], actions: List[int]) -> Tuple[list, list]:
        stack = [[]]
        for token, action in zip(tokens, actions):
            if action == 0:
                stack.append([])

            stack[-1].append(token)

            if action == 1:
                const = stack.pop()
                stack[-1].append(const)

        # TODO: Check this.
        return stack[-1], stack

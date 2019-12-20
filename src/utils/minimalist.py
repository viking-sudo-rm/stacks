from typing import List
from copy import copy


def from_merges(tokens: List[str], actions: List[int]) -> List[str]:
    stack = []
    buffer_idx = 0
    for action in actions:
        if action == 0:
            stack.append(tokens[buffer_idx])
            buffer_idx += 1
        elif len(stack) > 1:
            s1 = stack.pop()
            s2 = stack.pop()
            stack.append([s2, s1])

    return stack[-1], stack
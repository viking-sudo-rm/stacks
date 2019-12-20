from typing import List


def from_parentheses(tokens: List[str], actions: List[int]) -> List[str]:
    stack = [[]]
    for token, action in zip(tokens, actions):
        if action == 0:
            stack.append([])

        stack[-1].append(token)

        if action == 1:
            const = stack.pop()
            stack[-1].append(const)

    return stack[-1] if len(stack[-1]) > 1 else stack[-1][0], stack

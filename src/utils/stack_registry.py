from typing import Optional

from stacknn.superpos import Stack, NoOpStack, MultiPopStack, MultiPushStack


_STACKS = {
    "basic": Stack,
    "noop": NoOpStack,
    "kpop": MultiPopStack,
    "kpush": MultiPushStack,
}


def make_stack(stack_type_str: str,
               stack_dim: int,
               num_actions: Optional[int] = None,
               max_depth: Optional[int] = None):
    """Factory method for creating a stack from a type string."""
    stack_type = _STACKS[stack_type_str]
    if stack_type_str == "kpop" or stack_type_str == "kpush":
        return stack_type(stack_dim,
                          max_depth=max_depth,
                          num_actions=num_actions)
    else:
        assert num_actions == stack_type.get_num_actions()
        return stack_type(stack_dim, max_depth=max_depth)

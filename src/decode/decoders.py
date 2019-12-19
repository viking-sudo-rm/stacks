from typing import Callable, List
from math import log2
import numpy as np

from src.decode.states import AbstractDecoderState, DyckDecoderState, ExpressionDecoderState


def _get_initial_state(state_type):
    if state_type == "dyck":
        return DyckDecoderState.initial()
    elif state_type == "reduce":
        return ExpressionDecoderState.initial()
    else:
        raise ValueError("Unsupported state type.")


def beam_decode(probabilities: np.ndarray,
                state_type: str,
                get_action: Callable[[int], int] = lambda x: x,
                full: bool = True,
                top_k: int = 3,
                beam_size: int = 50) -> List[int]:
    """Do a beam search where we enforce counter constraints over the sequence."""
    # Construct a beam of decoder states.
    states = [_get_initial_state(state_type)]
    
    for idx in range(len(probabilities)):
        final = full and (idx == len(probabilities) - 1)

        # The probability vector for this time step.
        probs = probabilities[idx, :]
        # The actions we should consider taking.
        actions = np.argsort(probs)[-top_k:]

        # The beam from the previous index.
        prev_states = sorted(states[:beam_size], key=lambda state: -state.log_prob)
        # The new beam to build up.
        states = []

        for state in prev_states:
            for action_idx in actions:
                action_prob = probs[action_idx]
                action = get_action(action_idx)
                log_prob = log2(action_prob + np.finfo(float).eps)
                next_state = state.transition(idx, action, log_prob, final=final)
                if next_state is not None:
                    states.append(next_state)

    return states[0].get_actions() if states else None


def greedy_decode(probabilities: np.ndarray,
                  state_type: str,
                  get_action: Callable[[int], int] = lambda x: x,
                  full: bool = True) -> List[int]:
    """Take the highest probability option that doesn't violate the constraints."""
    state = _get_initial_state(state_type)
    for idx in range(len(probabilities)):
        final = full and (idx == len(probabilities) - 1)
        probs = probabilities[idx, :]
        actions = np.argsort(probs)
        for action_idx in reversed(actions):
            action_prob = probs[action_idx]
            action = get_action(action_idx)
            log_prob = log2(action_prob + np.finfo(float).eps)
            next_state = state.transition(idx, action, log_prob, final=final)
            if next_state is not None:
                state = next_state
                break
    return state.get_actions()

from typing import Callable, List
from math import log2
import numpy as np

from src.decode.states import MinimalistDecoderState


def beam_dmg_decode(probabilities: np.ndarray,
                    length: int,  # The length of the string.
                    get_action: Callable[[int], int] = lambda x: x,
                    top_k: int = 3,
                    beam_size: int = 50) -> List[int]:
    """Do a beam search where we enforce counter constraints over the sequence."""
    # Construct a beam of decoder states.
    states = [MinimalistDecoderState.initial()]
    
    for idx in range(len(probabilities)):
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
                next_state = state.transition(idx, action, log_prob, length=length)
                if next_state is not None:
                    states.append(next_state)

    return states[0].get_actions() if states else None


def greedy_dmg_decode(probabilities: np.ndarray,
                      length: int,  # The length of the string.
                      get_action: Callable[[int], int] = lambda x: x) -> List[int]:
    """Take the highest probability option that doesn't violate the constraints."""
    state = MinimalistDecoderState.initial()
    for idx in range(len(probabilities)):
        probs = probabilities[idx, :]
        actions = np.argsort(probs)
        for action_idx in reversed(actions):
            action_prob = probs[action_idx]
            action = get_action(action_idx)
            log_prob = log2(action_prob + np.finfo(float).eps)
            next_state = state.transition(idx, action, log_prob, length=length)
            if next_state is not None:
                state = next_state
                break
    return state.get_actions()

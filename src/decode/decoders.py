from typing import Callable, List
from math import log2
import numpy as np

from src.decode.states import DecoderState


_ActionGetter = Callable[[int], int]


def beam_decode(probabilities: np.ndarray,
                num_tokens: int,
                state_type: DecoderState,
                get_action: _ActionGetter = lambda x: x,
                enforce_full: bool = True,
                top_k: int = 10,
                beam_size: int = 50) -> List[int]:
    """Do a beam search where we enforce counter constraints over the sequence."""
    states = [state_type.initial()]  # Beam of decoder states.
    
    for idx in range(len(probabilities)):
        probs = probabilities[idx, :]
        actions = np.argsort(probs)[-top_k:]
        prev_states = sorted(states[:beam_size], key=lambda state: -state.log_prob)
        states = []

        for state in prev_states:
            for action_idx in actions:
                action_prob = probs[action_idx]
                action = get_action(action_idx)
                log_prob = log2(action_prob + np.finfo(float).eps)
                next_state = state.transition(idx, action, log_prob, num_tokens, enforce_full)
                if next_state is not None:
                    states.append(next_state)

    return states[0].get_actions() if states else None


def greedy_decode(probabilities: np.ndarray,
                  num_tokens: int,
                  state_type: DecoderState,
                  get_action: _ActionGetter = lambda x: x,
                  enforce_full: bool = True) -> List[int]:
    """Take the highest probability option that doesn't violate the constraints."""
    state = state_type.initial()

    for idx in range(len(probabilities)):
        probs = probabilities[idx, :]
        actions = np.argsort(probs)

        # Take the first action that is valid.
        next_state = None
        for action_idx in reversed(actions):
            action_prob = probs[action_idx]
            action = get_action(action_idx)
            log_prob = log2(action_prob + np.finfo(float).eps)
            next_state = state.transition(idx, action, log_prob, num_tokens, enforce_full)
            if next_state is not None:
                break

        # If there is no valid action, return null.
        if next_state is None:
            return None
        state = next_state

    return state.get_actions()

from overrides import overrides

from src.decode.states.base import AbstractDecoderState


class MinimalistDecoderState(AbstractDecoderState):

    def __init__(self,
                 action: int,
                 depth: int,
                 num_pushes: int,
                 num_merges: int,
                 log_prob: float,
                 parent):
        self.action = action
        self.depth = depth
        self.num_pushes = num_pushes
        self.num_merges = num_merges
        self.log_prob = log_prob
        self.parent = parent

    @classmethod
    def initial(cls):
        return cls(action=0, depth=0, num_pushes=0, num_merges=0, log_prob=0, parent=None)

    @overrides
    def transition(self, idx: int, action: int, action_log_prob: float, length: int):
        # Fixme: Ignore merges at zero depth?
        depth = self.depth + (1 if action == 0 else -1)
        num_pushes = self.num_pushes + int(action == 0)
        num_merges = self.num_merges + int(action == 1)

        if num_pushes > length:
            # Don't push more than the number of words.
            return None
        elif num_merges > length - 1:
            # Don't merge more than n - 1 times.
            return None

        log_prob = self.log_prob + action_log_prob
        return MinimalistDecoderState(action=action,
                                      depth=depth,
                                      num_pushes=num_pushes,
                                      num_merges=num_merges,
                                      log_prob=log_prob,
                                      parent=self)

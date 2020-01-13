from overrides import overrides

from src.decode.states.base import DecoderState


@DecoderState.register("merge")
class MergeDecoderState(DecoderState):

    """A decoder state that converts shift, merge sequences to binary parse trees."""

    def __init__(self,
                 action: int,
                 log_prob: float,
                 parent,
                 depth: int,
                 num_pushes: int,
                 num_merges: int):
        super().__init__(action, log_prob, parent)
        self.depth = depth
        self.num_pushes = num_pushes
        self.num_merges = num_merges

    @classmethod
    def initial(cls):
        return cls(action=0, log_prob=0, parent=None, depth=0, num_pushes=0, num_merges=0)

    @overrides
    def transition(self,
                   idx: int,  # The index of the current action.
                   action: int,  # The value of the current action.
                   action_log_prob: float,  # The log-probability of the current action.
                   length: int,  # The length of the sentence.
                   enforce_full: bool  # Constrain to only full trees.
                  ):
        depth = self.depth + (1 if action == 0 else -1)
        num_pushes = self.num_pushes + int(action == 0)
        num_merges = self.num_merges + int(action == 1)

        # Can't merge when there are less than two things on the stack.
        if depth < 2 and action == 1:
          return None

        # Can't push more than the number of words.
        if enforce_full and num_pushes > length:
            return None

        # Can't merge more than n - 1 times.
        if enforce_full and num_merges > length - 1:
            return None

        log_prob = self.log_prob + action_log_prob
        return MergeDecoderState(action=action,
                                 log_prob=log_prob + action_log_prob,
                                 parent=self,
                                 depth=depth,
                                 num_pushes=num_pushes,
                                 num_merges=num_merges)

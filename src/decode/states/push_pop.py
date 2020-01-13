from overrides import overrides

from src.decode.states.base import DecoderState


@DecoderState.register("push-pop")
class PushPopDecoderState(DecoderState):

    """A decoder state describing a parentheses-like parsing scheme where each word either opens or
    closes a constituent.
    """

    def __init__(self, action: int, log_prob: float, parent, depth: int):
        super().__init__(action, log_prob, parent)
        self.depth = depth
        # TODO: Should we add constraint in terms of num_pushes here?

    @classmethod
    def initial(cls):
        return cls(action=0, log_prob=0, parent=None, depth=0)

    @overrides
    def transition(self,
                   idx: int,  # The index of the current action.
                   action: int,  # The value of the current action.
                   action_log_prob: float,  # The log-probability of the current action.
                   length: int,  # The length of the sentence.
                   enforce_full: bool  # Constrain to only full trees.
                  ):
        next_depth = self.depth + (1 if action == 0 else -1)

        if next_depth < 0:
            return None

        if full_only and idx == length - 1 and next_depth != 0:
            return None

        return PushPopDecoderState(action=action,
                                   log_prob=self.log_prob + action_log_prob,
                                   parent=self,
                                   depth=next_depth)

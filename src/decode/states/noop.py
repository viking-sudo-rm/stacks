from overrides import overrides

from src.decode.states.base import DecoderState


@DecoderState.register("noop")
class NoOpDecoderState(DecoderState):

    """A decoder state describing a similar parsing scheme to push-pop, except that some words can
    now take no action.
    """

    _DEPTH_UPDATES = [1, 0, -1]

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
        next_depth = self.depth + self._DEPTH_UPDATES[action]

        if next_depth < 0:
            return None

        if enforce_full and idx == length - 1 and next_depth != 0:
            return None

        return NoOpDecoderState(action=action,
                                log_prob=self.log_prob + action_log_prob,
                                parent=self,
                                depth=next_depth)

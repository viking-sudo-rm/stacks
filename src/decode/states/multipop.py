from overrides import overrides

from src.decode.states.base import DecoderState


@DecoderState.register("kpop")
class MultipopDecoderState(DecoderState):

    """A decoder state describing an expression-like parsing scheme capable of closing multiple
    constituents per word.

    Action meanings: k=REDUCE(k) for 0<=k.
    """

    def __init__(self, action: int, log_prob: float, parent, depth: int):
        super().__init__(action, log_prob, parent)
        self.depth = depth

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
        depth = self.depth + 1 - action

        if depth < 1:
            return None

        if enforce_full and idx == length - 1 and depth != 1:
            return None

        return MultipopDecoderState(action=action,
                                    log_prob=self.log_prob + action_log_prob,
                                    parent=self,
                                    depth=depth)

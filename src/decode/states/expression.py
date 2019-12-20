from overrides import overrides

from src.decode.states.base import AbstractDecoderState


class ExpressionDecoderState(AbstractDecoderState):

    # Action nums: -1=INIT, k=REDUCE(k) for 0<=k

    @classmethod
    def initial(cls):
        return cls(action=0, counter=0, log_prob=0, parent=None)

    @overrides
    def transition(self, idx: int, action: int, action_log_prob: float, final: bool):
        next_counter = self.counter + 1 - action
        # FIXME: Perhaps think about how to deal with final transitions better.
        if not final and next_counter < 1:
            # We never want to reduce more than the number of constituents.
            return None
        elif final and next_counter != 1:
            # At the final configuration, we need exactly one constituent left.
            return None
        else:
            return ExpressionDecoderState(action=action,
                                          counter=next_counter,
                                          log_prob=self.log_prob + action_log_prob,
                                          parent=self)

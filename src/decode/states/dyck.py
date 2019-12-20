from overrides import overrides

from src.decode.states.base import AbstractDecoderState


class DyckDecoderState(AbstractDecoderState):

    @classmethod
    def initial(cls):
        return cls(action=0, counter=0, log_prob=0, parent=None)

    @overrides
    def transition(self, idx: int, action: int, action_log_prob: float, final: bool):
        next_counter = self.counter + (1 if action == 0 else -1)
        if not final and next_counter < 0:
            # We never want to reduce more than the number of constituents.
            return None
        elif final and next_counter != 0:
            # At a final configuration, we need exactly one constituent left to get a full tree.
            return None
        else:
            return DyckDecoderState(action=action,
                                    counter=next_counter,
                                    log_prob=self.log_prob + action_log_prob,
                                    parent=self)

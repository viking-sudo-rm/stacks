from abc import ABCMeta, abstractmethod
from overrides import overrides


class AbstractDecoderState(metaclass=ABCMeta):

    def __init__(self, action, counter, log_prob, parent):
        self.action = action
        self.counter = counter
        self.log_prob = log_prob
        self.parent = parent

    def get_actions(self):
        actions = []
        state = self
        # If state.parent is None, then we are in the initial state.
        while state.parent is not None:
            actions.insert(0, state.action)
            state = state.parent
        return actions

    @classmethod
    @abstractmethod
    def initial(cls):
        return NotImplemented

    @abstractmethod
    def transition(self, idx: int, action: int, action_log_prob: float, final: bool):
        return NotImplemented


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

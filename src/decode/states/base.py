from abc import ABCMeta, abstractmethod


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

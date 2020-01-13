from abc import ABCMeta, abstractmethod

from allennlp.common.registrable import Registrable


class DecoderState(Registrable, metaclass=ABCMeta):

    def __init__(self, action: int, log_prob: float, parent):
        self.action = action
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
    def transition(self,
                   idx: int,  # The index of the current action.
                   action: int,  # The value of the current action.
                   action_log_prob: float,  # The log-probability of the current action.
                   length: int,  # The length of the sentence.
                   enforce_full: bool  # Constrain to only full trees.
                  ):
        return NotImplemented

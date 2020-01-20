from abc import ABCMeta, abstractmethod
from typing import List, Tuple

from allennlp.common.registrable import Registrable


class ActionParser(Registrable, metaclass=ABCMeta):

    @abstractmethod
    def get_parse(self, tokens: List[str], actions: List[int]) -> Tuple[list, list]:
        return NotImplemented

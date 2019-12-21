from abc import ABCMeta, abstractmethod
import torch

from allennlp.common.registrable import Registrable
from allennlp.modules import FeedForward


class AbstractController(Registrable, torch.nn.Module, metaclass=ABCMeta):

    """Wrap a FeedForward module into a controller."""

    @abstractmethod
    def reset(self, batch_size: int, device: int):
        """This method resets the internal state of the controller."""
        return NotImplemented

    @abstractmethod
    def forward(self,
                inputs: torch.Tensor,
                summaries: torch.Tensor) -> torch.Tensor:
        return NotImplemented

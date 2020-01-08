from abc import abstractmethod
import torch

from allennlp.common.registrable import Registrable


class StackController(Registrable, torch.nn.Module):

    @abstractmethod
    def reset(self, batch_size: int, device: int):
        return NotImplemented

    @abstractmethod
    def forward(self, inputs: torch.Tensor, summaries: torch.Tensor) -> torch.Tensor:
        return NotImplemented

    @abstractmethod
    def get_input_dim(self) -> int:
        return NotImplemented

    @abstractmethod
    def get_summary_dim(self) -> int:
        return NotImplemented

    @abstractmethod
    def get_output_dim(self) -> int:
        return NotImplemented

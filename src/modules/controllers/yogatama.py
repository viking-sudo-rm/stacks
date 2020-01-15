from overrides import overrides
import torch

from src.modules.controllers.base import StackController


@StackController.register("yogatama-generic-rnn")
class YogatamaRnnCellController(StackController):

    """This controller network is slightly different than the one used by Suzgun, et al. (2019).

    The model has an additional linear transformation "hmap" applies to the states.
    """

    _RNN_TYPES = {
        "lstm": torch.nn.LSTMCell,
        "gru": torch.nn.GRUCell,
    }

    def __init__(self,
                 input_dim: int,
                 summary_dim: int,  # Size of the stack summary vector.
                 hidden_dim: int,
                 rnn_cell_type: str):
        super().__init__()
        self.input_dim = input_dim
        self.summary_dim = summary_dim
        self.hidden_dim = hidden_dim

        self.hmap = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.smap = torch.nn.Linear(summary_dim, hidden_dim, bias=False)

        rnn_cell_type = self._RNN_TYPES[rnn_cell_type]
        self.rnn_cell = rnn_cell_type(input_dim, hidden_dim)

    @overrides
    def reset(self, batch_size: int, device: int):
        self.states = torch.zeros(batch_size, self.hidden_dim, device=device)
        self.cells = torch.zeros(batch_size, self.hidden_dim, device=device)

    @overrides
    def forward(self, inputs, summaries):
        states = self.hmap(self.states) + self.smap(summaries)
        self.states, self.cells = self.rnn_cell(inputs, [states, self.cells])
        return self.states

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_summary_dim(self) -> int:
        return self.summary_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.hidden_dim

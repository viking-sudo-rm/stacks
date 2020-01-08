import torch
from overrides import overrides

from src.modules.controllers.base import StackController


@StackController.register("suzgun-rnn")
class SuzgunRnnController(StackController):

    def __init__(self,
                 input_dim: int,
                 summary_dim: int,  # Size of the stack summary vector.
                 hidden_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.input_dim = input_dim
        self.summary_dim = summary_dim
        self.hidden_dim = hidden_dim

        self.smap = torch.nn.Linear(summary_dim, hidden_dim, bias=False)
        self.imap = torch.nn.Linear(input_dim, hidden_dim)
        self.hmap = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dropout = torch.nn.Dropout(p=dropout)

    @overrides
    def reset(self, batch_size: int, device: int):
        self.states = torch.zeros(batch_size, self.hidden_dim, device=device)

    @overrides
    def forward(self, inputs, summaries):
        states = self.states + self.smap(summaries)
        self.states = torch.tanh(self.imap(inputs) + self.hmap(states))
        self.states = self.dropout(self.states)
        return self.states

    @overrides
    def get_input_dim(self):
        return self.input_dim

    @overrides
    def get_summary_dim(self):
        return self.summary_dim

    @overrides
    def get_output_dim(self):
        return self.hidden_dim


@StackController.register("suzgun-generic-rnn")
class SuzgunRnnCellController(StackController):

    """RNN controller for a complex RNN type, i.e. LSTM/GRU."""

    _RNN_TYPES = {
        "lstm": torch.nn.LSTMCell,
        "gru": torch.nn.GRUCell,
    }

    def __init__(self,
                 input_dim: int,
                 summary_dim: int,  # Size of the stack summary vector.
                 hidden_dim: int,
                 rnn_cell_type: str,  # torch.nn.LSTMCell or torch.nn.GRUCell.
                 dropout: float = 0.):
        super().__init__()
        self.input_dim = input_dim
        self.summary_dim = summary_dim
        self.hidden_dim = hidden_dim

        self.smap = torch.nn.Linear(summary_dim, hidden_dim, bias=False)
        self.dropout = torch.nn.Dropout(p=dropout)

        rnn_cell_type = self._RNN_TYPES[rnn_cell_type]
        self.rnn_cell = rnn_cell_type(input_dim, hidden_dim)

    @overrides
    def reset(self, batch_size: int, device: int):
        self.states = torch.zeros(batch_size, self.hidden_dim, device=device)
        self.cells = torch.zeros(batch_size, self.hidden_dim, device=device)

    @overrides
    def forward(self, inputs, summaries):
        states = self.states + self.smap(summaries)
        self.states, self.cells = self.rnn_cell(inputs, [states, self.cells])
        self.states = self.dropout(self.states)
        return self.states

    @overrides
    def get_input_dim(self):
        return self.input_dim

    @overrides
    def get_summary_dim(self):
        return self.summary_dim

    @overrides
    def get_output_dim(self):
        return self.hidden_dim

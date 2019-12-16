import torch


class SuzgunRnnController(torch.nn.Module):

    def __init__(self, input_dim: int, stack_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.smap = torch.nn.Linear(stack_dim, hidden_dim, bias=False)
        self.imap = torch.nn.Linear(input_dim, hidden_dim)
        self.hmap = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def reset(self, batch_size: int, device: int):
        self.states = torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, inputs, summaries):
        states = self.states + self.smap(summaries)
        self.states = torch.tanh(self.imap(inputs) + self.hmap(states))
        return self.states

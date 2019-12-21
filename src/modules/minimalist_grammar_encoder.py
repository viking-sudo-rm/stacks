from overrides import overrides
import torch

from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder

from stacknn.superpos import MinimalistStack
from stacknn.utils.expectation import get_expectation

from src.modules.controllers.rnn import SuzgunRnnController, SuzgunRnnCellController


def get_action(logits):
    dists = torch.softmax(logits, dim=-1)
    return get_expectation(dists)


@Seq2SeqEncoder.register("minimalist-grammar")
class MinimalistGrammarEncoder(Seq2SeqEncoder):

    SUMMARY_SIZE = 2

    def __init__(self,
                 input_dim: int,
                 stack_dim: int,
                 hidden_dim: int,
                 dropout: float = 0.,
                 store_policies: bool = False,
                 lstm_controller: bool = False):
        super().__init__()
        self.stack_dim = stack_dim
        self.summary_dim = self.SUMMARY_SIZE * stack_dim
        
        if lstm_controller:
            self.controller = SuzgunRnnCellController(input_dim, self.summary_dim, hidden_dim,
                                                      rnn_cell_type=torch.nn.LSTMCell,
                                                      dropout=dropout)
        else:
            self.controller = SuzgunRnnController(input_dim, self.summary_dim, hidden_dim,
                                                  dropout=dropout)

        # TODO: Replace these parameters with controller.get_input_dim(), etc.
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.bidirectional = False

        self.policy = torch.nn.Linear(self.output_dim, MinimalistStack.get_num_actions())
        self.vectorizer = torch.nn.Linear(self.output_dim, stack_dim)

        self.all_policies = None
        self.store_policies = store_policies

    @overrides
    def forward(self, inputs, mask):
        batch_size, seq_len, _ = inputs.size()
        mask = mask.float().unsqueeze(-1)  # Change shape for working with input distributions.

        # Initialize memory.
        stacks = MinimalistStack.empty(batch_size, self.stack_dim, device=inputs.device)
        summaries = torch.zeros(batch_size, self.summary_dim, device=inputs.device)
        self.controller.reset(batch_size, device=inputs.device)

        # Initialize the model to be looking at the first word.
        input_dists = torch.zeros_like(mask)
        input_dists[:, 0, :] = 1.

        all_states = []
        self.all_policies = []
        for _ in range(2 * seq_len - 1):
            # Update the controller state.
            superpos_inputs = torch.sum(input_dists * inputs, dim=1)
            states = self.controller(superpos_inputs, summaries)
            all_states.append(states)

            # Compute a distribution over actions to take.
            policies = torch.softmax(self.policy(states), dim=-1)
            vectors = torch.sigmoid(self.vectorizer(states))
            stacks.update(policies, vectors)

            # Compute the summary of the modified stack.
            tapes = stacks.tapes[:, :self.SUMMARY_SIZE, :]
            length = tapes.size(1)
            if length < self.SUMMARY_SIZE:
                # If necessary, we pad the summaries with zeros.
                summaries = torch.zeros(batch_size, self.SUMMARY_SIZE, self.stack_dim,
                                        device=inputs.device)
                summaries[:, :length, :] = tapes
            summaries = torch.flatten(summaries, start_dim=1)

            # Update the position distribution over the input.
            input_dists = self._shift_input(input_dists, policies, mask)

            # Optionally store the policies at inference time.
            if self.store_policies:
                self.all_policies.append(policies)

        # Optionally store the policies at inference time.
        if self.store_policies:
            self.all_policies = torch.stack(self.all_policies, dim=1)

        return torch.stack(all_states, dim=1)

    def _shift_input(self, input_dists, policies, mask):
        """Compute the next input distribution as a superposition of shifting and merging."""
        push_dists = torch.zeros_like(input_dists)
        push_dists[:, 1:, :] = input_dists[:, :-1, :]
        push_dists = push_dists * mask
        # Pushing shifts the input; merging preserves the current position.
        policies = policies.unsqueeze(-1).unsqueeze(-1)
        return policies[:, 0] * push_dists + policies[:, 1] * input_dists

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return  self.output_dim

    @overrides
    def is_bidirectional(self) -> bool:
        return self.bidirectional

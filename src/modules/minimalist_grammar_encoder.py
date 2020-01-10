from overrides import overrides
import torch

from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder

from stacknn.superpos import MinimalistStack
from stacknn.utils.expectation import get_expectation

from src.modules.controllers import StackController


def get_action(logits):
    dists = torch.softmax(logits, dim=-1)
    return get_expectation(dists)


@Seq2SeqEncoder.register("minimalist-grammar")
class MinimalistGrammarEncoder(Seq2SeqEncoder):

    def __init__(self,
                 stack_dim: int,
                 summary_size: int,
                 controller: StackController,
                 store_policies: bool = False):
        super().__init__()        
        self.controller = controller
        self.stack_dim = stack_dim
        self.summary_size = summary_size
        self.summary_dim = self.controller.get_summary_dim()
        # With the superposition pooling, this model can look ahead.
        self.bidirectional = True

        # Make sure the encoder stack size matches the controller summary size.
        assert self.stack_dim * self.summary_size == self.summary_dim

        output_dim = self.get_output_dim()
        self.policy = torch.nn.Linear(output_dim, MinimalistStack.get_num_actions())
        self.vectorizer = torch.nn.Linear(output_dim, stack_dim)

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

        all_input_dists = []
        all_states = []
        self.all_policies = []
        for _ in range(2 * seq_len - 1):
            # Update the controller state.
            superpos_inputs = torch.sum(input_dists * inputs, dim=1)
            states = self.controller(superpos_inputs, summaries)
            all_input_dists.append(input_dists)
            all_states.append(states)

            # Compute a distribution over actions to take.
            policies = torch.softmax(self.policy(states), dim=-1)
            vectors = torch.sigmoid(self.vectorizer(states))
            stacks.update(policies, vectors)

            # Update the stack summary and input distribution.
            summary = self._summarize(stacks)
            input_dists = self._shift_input(input_dists, policies, mask)

            # Optionally store the policies at inference time.
            if self.store_policies:
                self.all_policies.append(policies)

        # Optionally store the policies at inference time.
        if self.store_policies:
            self.all_policies = torch.stack(self.all_policies, dim=1)

        return self._get_superpos_states(all_input_dists, all_states)

    def _shift_input(self, input_dists, policies, mask):
        """Compute the distribution for the next input position."""
        push_dists = torch.zeros_like(input_dists)
        push_dists[:, 1:, :] = input_dists[:, :-1, :]
        push_dists = push_dists * mask
        # Pushing shifts the input; merging preserves the current position.
        policies = policies.unsqueeze(-1).unsqueeze(-1)
        return policies[:, 0] * push_dists + policies[:, 1] * input_dists

    def _get_superpos_states(self, all_input_dists, all_states):
        """Return a superimposed output sequence of the same length as input."""
        all_input_dists = torch.stack(all_input_dists, dim=2)  # [*, seq_len, num_steps, 1].
        all_input_dists = all_input_dists.squeeze(-1)  # [*, seq_len, num_steps].
        all_states = torch.stack(all_states, dim=1)  # [*, num_steps, hidden_dim].
        return all_input_dists @ all_states

    # TODO: Add this as utility function to StackNN-Core.
    def _summarize(self, stack):
        """Compute the summary of the modified stack."""
        summary = stack.tapes[:, :self.summary_size, :]
        batch_size, length, stack_dim = summary.size()

        if length < self.summary_size:
            device = summary.device
            padded_summary = torch.zeros(batch_size, self.summary_size, stack_dim, device=device)
            padded_summary[:, :length, :] = summary
            summary = padded_summary

        return torch.flatten(summary, start_dim=1)

    @overrides
    def get_input_dim(self) -> int:
        return self.controller.get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return  self.controller.get_output_dim()

    @overrides
    def is_bidirectional(self) -> bool:
        return self.bidirectional

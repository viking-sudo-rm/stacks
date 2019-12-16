from overrides import overrides
import torch

from allennlp.modules import FeedForward
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder

from stacknn.superpos import Stack
from stacknn.utils.expectation import get_expectation

from src.modules.suzgun_controller import SuzgunRnnController


def get_action(logits):
    dists = torch.softmax(logits, dim=-1)
    return get_expectation(dists)


@Seq2SeqEncoder.register("stack-encoder")
class StackEncoder(Seq2SeqEncoder):

    def __init__(self,
                 input_dim: int,
                 stack_dim: int,
                 hidden_dim: int,
                 store_policies: bool = False):
        super().__init__()
        self.stack_dim = stack_dim
        self.stack_type = Stack
        self.controller = SuzgunRnnController(input_dim, stack_dim, hidden_dim)

        # TODO: Replace these parameters with controller.get_input_dim(), etc.
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.bidirectional = False

        self.policy = torch.nn.Linear(self.output_dim, self.stack_type.get_num_actions())
        self.vectorizer = torch.nn.Linear(self.output_dim, stack_dim)

        self.all_policies = None
        self.store_policies = store_policies

    @overrides
    def forward(self, inputs, mask):
        batch_size, seq_len, _ = inputs.size()
        stacks = self.stack_type.empty(batch_size, self.stack_dim, device=inputs.device)
        summaries = torch.zeros(batch_size, self.stack_dim, device=inputs.device)
        self.controller.reset(batch_size, device=inputs.device)

        all_states = []
        self.all_policies = []
        for time_idx in range(seq_len):
            # Update the controller state.
            states = self.controller(inputs[:, time_idx, :], summaries)
            all_states.append(states)

            # Compute a distribution over actions to take.
            policies = torch.softmax(self.policy(states), dim=-1)
            vectors = torch.sigmoid(self.vectorizer(states))

            # Update the stack and compute the next summary.
            stacks.update(policies, vectors)
            summaries = stacks.tapes[:, 0, :]

            # Optionally store the policies at inference time.
            if self.store_policies:
                self.all_policies.append(policies)

        # Optionally store the policies at inference time.
        if self.store_policies:
            self.all_policies = torch.stack(self.all_policies, dim=1)

        return torch.stack(all_states, dim=1)

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return  self.output_dim

    @overrides
    def is_bidirectional(self) -> bool:
        return self.bidirectional

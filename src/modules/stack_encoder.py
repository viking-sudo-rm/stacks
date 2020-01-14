from overrides import overrides
import torch

from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder

from stacknn.superpos import Stack, NoOpStack, MultiPopStack, MultiPushStack
from stacknn.utils.expectation import get_expectation

from src.modules.controllers import StackController

# _STACK_TYPES = {
#     "basic": Stack,
#     "noop": NoOpStack,
#     "multipop": MultiPopStack,
# }


def get_action(logits):
    dists = torch.softmax(logits, dim=-1)
    return get_expectation(dists)


@Seq2SeqEncoder.register("stack-encoder")
class StackEncoder(Seq2SeqEncoder):

    def __init__(self,
                 stack_dim: int,
                 summary_size: int,
                 controller: StackController,
                 num_actions: int = 6,
                 store_policies: bool = False,
                 project_states: bool = True,
                 multipush: bool = False):
        super().__init__()
        self.stack_dim = stack_dim
        self.summary_size = summary_size
        self.summary_dim = summary_size * stack_dim

        self.controller = controller
        stack_type = MultiPopStack if not multipush else MultiPushStack
        self.stacks = stack_type(stack_dim, num_actions)

        self.all_policies = None
        self.store_policies = store_policies

        output_dim = self.get_output_dim()
        self.policy = torch.nn.Linear(output_dim, num_actions)

        if project_states:
            self.vectorizer = torch.nn.Linear(output_dim, stack_dim)
        else:
            assert output_dim == stack_dim
            self.vectorizer = None

    @overrides
    def forward(self, inputs, mask):
        batch_size, seq_len, _ = inputs.size()
        self.stacks.reset(batch_size, device=inputs.device)
        self.controller.reset(batch_size, device=inputs.device)
        summaries = torch.zeros(batch_size, self.summary_dim, device=inputs.device)

        all_states = []
        self.all_policies = []
        for time_idx in range(seq_len):
            states = self.controller(inputs[:, time_idx, :], summaries)
            all_states.append(states)
            policies = torch.softmax(self.policy(states), dim=-1)
            # FIXME: Is sigmoid okay here? Would tanh make a difference?
            vectors = torch.sigmoid(self.vectorizer(states)) if self.vectorizer else states

            # Update the stack and compute the next summary.
            self.stacks.update(policies, vectors)
            if self.summary_size == 1:
                summaries = self.stacks.tapes[:, 0, :]
            else:
                tapes = self.stacks.tapes[:, :self.summary_size, :]
                length = tapes.size(1)
                if length < self.summary_size:
                    # If necessary, we pad the summaries with zeros.
                    summaries = torch.zeros(batch_size, self.summary_size, self.stack_dim,
                                            device=inputs.device)
                    summaries[:, :length, :] = tapes
                summaries = torch.flatten(summaries, start_dim=1)

            # Optionally store the policies at inference time.
            if self.store_policies:
                self.all_policies.append(policies)

        # Optionally store the policies at inference time.
        if self.store_policies:
            self.all_policies = torch.stack(self.all_policies, dim=1)

        return torch.stack(all_states, dim=1)

    @overrides
    def get_input_dim(self) -> int:
        return self.controller.get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return  self.controller.get_output_dim()

    @overrides
    def is_bidirectional(self) -> bool:
        return False

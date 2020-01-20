from overrides import overrides
import torch
from typing import Optional

from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder

from stacknn.superpos import Stack, NoOpStack, MultiPopStack, MultiPushStack
from stacknn.utils.expectation import get_expectation

from src.modules.controllers import StackController
from src.utils.stack_registry import make_stack


def get_action(logits):
    dists = torch.softmax(logits, dim=-1)
    return get_expectation(dists)


@Seq2SeqEncoder.register("stack-encoder")
class StackEncoder(Seq2SeqEncoder):

    def __init__(self,
                 stack_dim: int,
                 summary_size: int,
                 controller: StackController,
                 stack_type: str = "kpop",
                 num_actions: int = 6,
                 max_depth: Optional[int] = None,
                 dropout: float = 0.,  # Dropout applies to the controller states.
                 store_policies: bool = False,
                 project_states: bool = True):
        super().__init__()
        self.stack_dim = stack_dim
        self.summary_size = summary_size
        self.summary_dim = summary_size * stack_dim
        self.num_actions = num_actions

        self.controller = controller
        self.output_dim = self.get_output_dim()
        # TODO: Replace this with a factory or something?
        self.stacks = make_stack(stack_type, stack_dim,
                                 num_actions=num_actions,
                                 max_depth=max_depth)

        self.all_policies = None
        self.store_policies = store_policies
        self.policy = torch.nn.Linear(self.output_dim, num_actions)

        if project_states:
            self.vectorizer = torch.nn.Linear(self.output_dim, stack_dim)
        else:
            assert self.output_dim == stack_dim, \
                "Without a state projection, output_dim and stack_dim need to match."
            self.vectorizer = None

        self.dropout = torch.nn.Dropout(dropout)

    @overrides
    def forward(self, inputs, mask):
        batch_size, seq_len, _ = inputs.size()
        device = inputs.device
        self.stacks.reset(batch_size, device=device)
        self.controller.reset(batch_size, device=device)
        summaries = torch.zeros(batch_size, self.summary_dim, device=device)

        all_states = torch.empty(batch_size, seq_len, self.output_dim, device=device)
        if self.store_policies:
            self.all_policies = torch.empty(batch_size, seq_len, self.num_actions, device=device)

        for time_idx in range(seq_len):
            states = self.controller(inputs[:, time_idx, :], summaries)
            states = self.dropout(states)
            all_states[:, time_idx, :] = states

            policies = torch.softmax(self.policy(states), dim=-1)
            # We use sigmoid here. Would tanh make a difference?
            vectors = torch.sigmoid(self.vectorizer(states)) if self.vectorizer else states
            self.stacks.update(policies, vectors)

            # Compute the next summary.
            if self.summary_size == 1:
                summaries = self.stacks.tapes[:, 0, :]
            else:
                tapes = self.stacks.tapes[:, :self.summary_size, :]
                length = tapes.size(1)
                if length < self.summary_size:
                    # If necessary, we pad the summaries with zeros.
                    summaries = torch.zeros(batch_size, self.summary_size, self.stack_dim,
                                            device=device)
                    summaries[:, :length, :] = tapes
                summaries = torch.flatten(summaries, start_dim=1)

            # Optionally store the policies at inference time.
            if self.store_policies:
                self.all_policies[:, time_idx, :] = policies

        return all_states

    @overrides
    def get_input_dim(self) -> int:
        return self.controller.get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return  self.controller.get_output_dim()

    @overrides
    def is_bidirectional(self) -> bool:
        return False

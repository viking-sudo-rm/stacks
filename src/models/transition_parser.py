from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy

from stacknn.superpos import TransitionParserStack

from src.modules.controllers import StackController


@Model.register("transition-parser")
class TransitionParser(Model):

    # TODO: Add ability to have hard/straight-through stack.

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 controller: StackController,
                 summary_size: int,
                 hard: bool = False):

        super().__init__(vocab)
        self.embedder = text_field_embedder
        self.controller = controller
        self.accuracy = CategoricalAccuracy()

        self.stack_dim = self.embedder.get_output_dim()
        self.summary_size = summary_size
        self.summary_dim = controller.get_summary_dim()
        assert self.stack_dim * summary_size == self.summary_dim

        output_dim = controller.get_output_dim()
        num_actions = TransitionParserStack.get_num_actions()
        self.policy = torch.nn.Linear(output_dim, num_actions)
        self.hard = hard

        # We use this linear transformation to align stack action space to label action space.
        self.aligner = torch.nn.Linear(num_actions, num_actions)

    @overrides
    def forward(self, tokens, actions):
        batch_size, num_actions = actions.size()
        actions, act_mask = self._get_actions_and_mask(actions)
        buf_mask = self._get_buffer_mask(tokens)

        buf = self.embedder(tokens)
        stack = TransitionParserStack.empty(batch_size, self.stack_dim, device=buf.device)
        summary = torch.zeros(batch_size, self.summary_dim, device=buf.device)
        self.controller.reset(batch_size, device=buf.device)

        buf_pos = self._init_buffer_pos(buf_mask)
        policies = []

        for _ in range(num_actions):  # num_actions = 2 * num_tokens - 1.
            
            # Compute the policy using the controller.
            buf_head = torch.squeeze(buf_pos @ buf, dim=1)
            state = self.controller(buf_head, summary)
            policy = torch.softmax(self.policy(state), dim=-1)
            policies.append(policy)

            # Update the stack, buffer, and summary.
            stack_policy = self._get_stack_policy(policy)
            buf_pos = self._shift_buffer_pos(buf_pos, stack_policy, buf_mask)
            stack.update(stack_policy, buf_head)
            summary = self._summarize(stack)

        # Record the actions in the stack space and then map to label space.
        policies = torch.stack(policies, dim=1)
        output_dict = {"policies": policies}
        policies = self.aligner(policies)

        if actions is not None:
            loss = sequence_cross_entropy_with_logits(policies, actions, act_mask)
            output_dict["loss"] = loss
            self.accuracy(policies, actions, act_mask)

        return output_dict

    def _get_actions_and_mask(self, actions):
        actions = actions - 2.  # Remove the padding/OOV spots.
        actions = torch.where(actions < 0, torch.zeros_like(actions), actions)
        act_mask = (actions >= 0).float()
        return actions, act_mask

    def _get_buffer_mask(self, tokens):
        return get_text_field_mask(tokens).float().unsqueeze(1)

    def _get_stack_policy(self, policy):
        if self.hard:
            indices = torch.argmax(policy, dim=-1)
            return torch.nn.functional.one_hot(indices, policy.size(1))
        else:
            return policy

    def _init_buffer_pos(self, buf_mask):
        """Create the distribution of our position in the buffer."""
        buffer_pos = torch.zeros_like(buf_mask)
        buffer_pos[:, :, 0] = 1.
        return buffer_pos

    def _shift_buffer_pos(self, buf_pos, policy, buf_mask):
        """Compute the distribution for the next buffer position."""
        push_pos = torch.zeros_like(buf_pos)
        push_pos[:, :, 1:] = buf_pos[:, :, :-1]
        push_pos = push_pos * buf_mask
        policy = policy.unsqueeze(-1).unsqueeze(-1)
        # Left and right-arcs do not change the buffer; shifting does.
        return (policy[:, 0] + policy[:, 1]) * buf_pos + policy[:, 2] * push_pos

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
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

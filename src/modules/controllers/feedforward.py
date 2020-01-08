import torch
from overrides import overrides

from allennlp.modules.feedforward import FeedForward

from src.modules.controllers.base import StackController


@StackController.register("feedforward")
class FeedForwardController(StackController):

    def __init__(self,
                 input_dim: int,
                 summary_dim: int,
                 feedforward: FeedForward):
        super().__init__()
        self.input_dim = input_dim
        self.summary_dim = summary_dim
        self.feedforward = feedforward

        # Make sure that the input dimension matches the input/stack.
        assert input_dim + summary_dim == feedforward.get_input_dim()

    @overrides
    def reset(self, batch_size: int, device: int):
        pass

    @overrides
    def forward(self, inputs, summaries):
        features = torch.cat([inputs, summaries], dim=-1)
        return self.feedforward(features)

    @overrides
    def get_input_dim(self):
        return self.input_dim

    @overrides
    def get_summary_dim(self):
        return self.summary_dim

    @overrides
    def get_output_dim(self):
        return self.feedforward.get_output_dim()

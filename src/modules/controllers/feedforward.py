from overrides import overrides
import torch

from allennlp.modules import FeedForward

from src.modules.controllers.base import AbstractController


@AbstractController.register("feedforward")
class FeedForwardController(AbstractController):

    """Wrap a FeedForward module into a controller."""

    def __init__(self, feed_forward: FeedForward):
        super().__init__()
        self.feed_forward = feed_forward

    @overrides
    def reset(self, batch_size: int, device: int):
        pass

    @overrides
    def forward(self, inputs, summaries):
        features = torch.cat([inputs, summaries], dim=-1)
        return self.feed_forward(features)

import torch

from allennlp.models import LanguageModel, BasicClassifier, SimpleTagger

from src.modules.compose import ComposeEncoder


class PolicyListener:

    def __init__(self, policies_name: str):
        self.policies: torch.Tensor = None
        self.policies_name = policies_name

    def forward_hook(self, stack_encoder, inputs, outputs):
        self.policies = getattr(stack_encoder, self.policies_name)


def get_policies(model, instances, policies_name: str = "all_policies"):
    """Get the action dists from a model."""

    # We support multiple model types.
    if isinstance(model, LanguageModel):
        encoder = model._contextualizer
    elif isinstance(model, BasicClassifier):
        encoder = model._seq2seq_encoder
    elif isinstance(model, SimpleTagger):
        encoder = model.encoder
    else:
        raise ValueError("Unsupported model type.")

    # For multiple encoders, assume we are composing a contextualizer with a final stack layer.
    if isinstance(encoder, ComposeEncoder):
        encoder = encoder.encoders[-1]

    listener = PolicyListener(policies_name)
    encoder.store_policies = True
    encoder.register_forward_hook(listener.forward_hook)
    model.forward_on_instances(instances)
    return listener.policies.cpu().numpy()

import torch

from src.modules.compose import ComposeEncoder


_ENCODER_NAMES = ["_contextualizer", "_seq2seq_encoder", "encoder"]


class PolicyListener:

    def __init__(self, policies_name: str):
        self.policies: torch.Tensor = None
        self.policies_name = policies_name

    def forward_hook(self, stack_encoder, inputs, outputs):
        self.policies = getattr(stack_encoder, self.policies_name)


def get_policies(model, instances, policies_name: str = "all_policies"):
    """Get the action dists from a model."""

    # We support multiple model types.
    encoder = None
    for name in _ENCODER_NAMES:
        encoder = getattr(model, name, None)
        if encoder is not None:
            break
    if encoder is None:
        raise ValueError("Unsupported model.")

    # For multiple encoders, assume we are composing a contextualizer with a final stack layer.
    if isinstance(encoder, ComposeEncoder):
        encoder = encoder.encoders[-1]

    listener = PolicyListener(policies_name)
    encoder.store_policies = True
    encoder.register_forward_hook(listener.forward_hook)
    model.forward_on_instances(instances)
    return listener.policies.cpu().numpy()

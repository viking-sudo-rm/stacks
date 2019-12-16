import torch


class PolicyListener:

    def __init__(self, policies_name: str):
        self.policies: torch.Tensor = None
        self.policies_name = policies_name

    def forward_hook(self, stack_encoder, inputs, outputs):
        self.policies = getattr(stack_encoder, self.policies_name)


def get_policies(model, instances, policies_name: str = "all_policies"):
    """Get the action dists from a model."""
    model._contextualizer.store_policies = True
    listener = PolicyListener(policies_name)
    model._contextualizer.register_forward_hook(listener.forward_hook)
    model.forward_on_instances(instances)
    return listener.policies.cpu().numpy()

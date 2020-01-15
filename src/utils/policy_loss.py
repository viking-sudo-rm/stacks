import torch


def get_expected_num_pops(policies: torch.FloatTensor,
                          mask: torch.ByteTensor) -> torch.FloatTensor:
    """Compute the expected number of pops for each sequence in the batch."""
    num_actions = policies.size(-1)
    actions = torch.arange(0, num_actions, device=policies.device)
    actions = actions.unsqueeze(0).unsqueeze(0)

    exp_actions = torch.sum(policies * actions, dim=-1)
    return torch.sum(exp_actions * mask, dim=-1)


def get_variational_loss(policies: torch.FloatTensor,  # [*, seq_len, num_actions].
                         prior_distribution: torch.FloatTensor,  # [num_actions].
                        ) -> torch.FloatTensor:
    q = policies
    p = prior_distribution
    kl_divs = (q * (q / p).log()).sum(dim=-1)
    return torch.mean(kl_divs)

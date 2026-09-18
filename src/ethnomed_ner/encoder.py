import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dimension, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, value):
        normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, True) + self.eps)
        return (normalized * self.weight.float()).to(value.dtype)


def validate_mask(mask):
    if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[1] == 0:
        raise ValueError("Expected a nonempty [batch, length] boolean mask")
    if not mask[:, 0].all() or (mask[:, 1:] & ~mask[:, :-1]).any():
        raise ValueError("Every sequence must be nonempty and right padded")


def reverse_valid_tokens(value, mask):
    """Reverse content only; moving padding to the front contaminates the backward SSM."""
    validate_mask(mask)
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    indices = torch.where(mask, mask.sum(1, keepdim=True) - 1 - positions, positions)
    return value.gather(1, indices.unsqueeze(-1).expand_as(value))


class BidirectionalMamba3(nn.Module):
    """Table 6 bidirectional wrapper with a caller-supplied Mamba-3 factory.

    mixer_factory(config) must return a fresh causal Mamba-3 MIMO module mapping
    [batch, length, model_dim] to the same shape, without an outer residual.
    The factory is called independently for the two directions.
    """

    def __init__(self, config, mixer_factory=None):
        super().__init__()
        if mixer_factory is None:
            raise ValueError(
                "mixer_factory must construct a fresh causal Mamba-3 MIMO module"
            )
        self.forward_mixer = mixer_factory(config)
        self.backward_mixer = mixer_factory(config)
        if self.forward_mixer is self.backward_mixer:
            raise ValueError("mixer_factory must create independent directional modules")
        self.forward_norm = RMSNorm(config.model_dim)
        self.backward_norm = RMSNorm(config.model_dim)

    def forward(self, inputs, mask):
        validate_mask(mask)
        inputs = inputs.masked_fill(~mask.unsqueeze(-1), 0)
        forward = inputs + self.forward_mixer(self.forward_norm(inputs))
        reverse = reverse_valid_tokens(inputs, mask)
        backward = reverse + self.backward_mixer(self.backward_norm(reverse))
        backward = reverse_valid_tokens(backward, mask)
        return torch.cat((forward, backward), dim=-1).masked_fill(~mask.unsqueeze(-1), 0)

import torch
from torch import nn


def initialize_linear(layer):
    nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class BoundaryPrior(nn.Module):
    """Equations (4)-(5): a differentiable BIO prior predicted only from input states."""

    def __init__(self, backbone_dim=768, model_dim=256, boundary_dim=32):
        super().__init__()
        self.bottleneck = nn.Linear(backbone_dim, model_dim)
        self.bio_head = nn.Linear(model_dim, 3)
        self.bio_embedding = nn.Parameter(torch.empty(3, boundary_dim))
        self.input_projection = nn.Linear(model_dim + boundary_dim, model_dim)
        for layer in (self.bottleneck, self.bio_head, self.input_projection):
            initialize_linear(layer)
        nn.init.xavier_normal_(self.bio_embedding)

    def forward(self, hidden):
        features = self.bottleneck(hidden)
        bio_logits = self.bio_head(features)
        probabilities = bio_logits.float().softmax(dim=-1).to(features.dtype)
        prior = probabilities @ self.bio_embedding.to(features.dtype)
        inputs = self.input_projection(torch.cat((features, prior), dim=-1))
        return inputs, bio_logits

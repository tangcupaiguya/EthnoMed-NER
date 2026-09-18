from dataclasses import dataclass

import torch
from torch import nn
from torchcrf import CRF

from .boundary import BoundaryPrior, initialize_linear
from .data import decode_spans, encode_windows
from .encoder import BidirectionalMamba3, validate_mask
from .labels import LABELS


@dataclass
class NEROutput:
    hidden: torch.Tensor
    logits: torch.Tensor
    bio_logits: torch.Tensor | None = None


class BADP(nn.Module):
    def __init__(self, backbone, config, mixer_factory=None):
        super().__init__()
        self.config = config
        self.backbone = backbone
        if backbone.config.hidden_size != config.backbone_dim:
            raise ValueError("Student backbone hidden size does not match backbone_dim")
        self.boundary = BoundaryPrior(config.backbone_dim, config.model_dim, config.boundary_dim)
        self.encoder = BidirectionalMamba3(config, mixer_factory)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(2 * config.model_dim, len(LABELS))
        self.state_projection = nn.Linear(2 * config.model_dim, config.teacher_dim)
        self.crf = CRF(len(LABELS), batch_first=True)
        for layer in (self.classifier, self.state_projection):
            initialize_linear(layer)
        self.stage = None

    def set_stage(self, stage):
        """Apply Algorithm 1 freezing; construct the optimizer after each stage switch."""
        if stage not in (1, 2):
            raise ValueError("BADP stage must be 1 or 2")
        if self.config.lora:
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad_("lora_" in name)
        else:
            self.backbone.requires_grad_(True)
            self.backbone.get_input_embeddings().requires_grad_(stage == 2)
        self.state_projection.requires_grad_(stage == 1)
        self.crf.requires_grad_(stage == 2)
        # Optimizers can still update frozen parameters when an old gradient remains.
        for parameter in self.parameters():
            if not parameter.requires_grad:
                parameter.grad = None
        self.stage = stage
        return self

    def forward(self, batch, *, window_batch_size=None):
        hidden = encode_windows(self.backbone, batch, window_batch_size=window_batch_size)
        inputs, bio_logits = self.boundary(hidden)
        states = self.encoder(inputs, batch["mask"])
        return NEROutput(states, self.classifier(self.dropout(states)), bio_logits)

    def crf_loss(self, logits, labels, mask):
        validate_mask(mask)
        # torchcrf indexes all tags, even at masked positions; sanitize -100 padding first.
        tags = labels.masked_fill(~mask, 0)
        return -self.crf(logits.float(), tags, mask=mask, reduction="mean")

    @torch.no_grad()
    def decode(self, output, mask):
        validate_mask(mask)
        return self.crf.decode(output.logits.float(), mask=mask)

    @torch.inference_mode()
    def predict(self, batch, *, window_batch_size=None):
        """Return one list of character-offset Span objects per document."""
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            output = self(batch, window_batch_size=window_batch_size)
            tags = self.decode(output, batch["mask"])
            if len(tags) != len(batch["records"]):
                raise ValueError("Each decoded sequence must have a document record")
            return [
                decode_spans(sequence, record["offsets"])
                for sequence, record in zip(tags, batch["records"])
            ]
        finally:
            for module, training in modes:
                module.training = training


class NERTeacher(nn.Module):
    def __init__(self, backbone, hidden_size=768):
        super().__init__()
        if backbone.config.hidden_size != hidden_size:
            raise ValueError("Teacher backbone hidden size does not match teacher_dim")
        self.backbone = backbone
        self.classifier = nn.Linear(hidden_size, len(LABELS))
        initialize_linear(self.classifier)
        self._frozen = False

    def forward(self, batch, *, window_batch_size=None):
        hidden = encode_windows(self.backbone, batch, window_batch_size=window_batch_size)
        return NEROutput(hidden, self.classifier(hidden))

    def freeze(self):
        self._frozen = True
        self.requires_grad_(False)
        self.zero_grad(set_to_none=True)
        self.eval()
        return self

    def train(self, mode=True):
        """Keep a frozen distillation teacher in evaluation mode."""
        return super().train(False if self._frozen else mode)


def load_backbone(name, dropout=0.1):
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(name)
    if hasattr(config, "hidden_dropout_prob"):
        config.hidden_dropout_prob = dropout
    if hasattr(config, "attention_probs_dropout_prob"):
        config.attention_probs_dropout_prob = dropout
    return AutoModel.from_pretrained(name, config=config, add_pooling_layer=False)


def add_lora(backbone, config):
    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        backbone,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=[
                "attention.self.query",
                "attention.self.key",
                "attention.self.value",
                "attention.output.dense",
            ],
        ),
    )

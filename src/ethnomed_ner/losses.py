import torch
from torch import nn
from torch.nn import functional as F

from .labels import LABEL_TO_BIO, LABELS


def _check_mask(mask):
    if mask.dtype != torch.bool or not mask.any():
        raise ValueError("Losses require at least one valid token and a boolean mask")


def masked_cross_entropy(logits, labels, mask):
    _check_mask(mask)
    return F.cross_entropy(logits[mask].float(), labels[mask])


def masked_focal_loss(logits, labels, mask, gamma=2.0, alpha=None):
    """Equation (12), averaged over valid tokens, with no padding/special-token labels."""
    _check_mask(mask)
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    gold = labels[mask]
    log_probability = F.log_softmax(logits[mask].float(), dim=-1).gather(1, gold[:, None])[:, 0]
    loss = -(1 - log_probability.exp()).pow(gamma) * log_probability
    if alpha is not None:
        weights = torch.as_tensor(alpha, device=logits.device, dtype=torch.float32)
        if weights.shape != (logits.shape[-1],) or (weights < 0).any():
            raise ValueError("alpha must contain one nonnegative coefficient per label")
        loss = loss * weights[gold]
    return loss.mean()


def masked_distillation(student_logits, teacher_logits, mask, temperature=3.0):
    """Equation (9): KL(teacher || student); temperature squared appears exactly once."""
    _check_mask(mask)
    if temperature <= 0 or student_logits.shape != teacher_logits.shape:
        raise ValueError("Invalid temperature or unaligned teacher/student logits")
    student = F.log_softmax(student_logits[mask].float() / temperature, dim=-1)
    teacher = F.softmax(teacher_logits[mask].detach().float() / temperature, dim=-1)
    return F.kl_div(student, teacher, reduction="none").sum(-1).mean() * temperature**2


def masked_state_mse(projected_student, teacher_hidden, mask):
    _check_mask(mask)
    if projected_student.shape != teacher_hidden.shape:
        raise ValueError("Teacher and projected student states must have identical shapes")
    return F.mse_loss(projected_student[mask].float(), teacher_hidden[mask].detach().float())


class BADPObjective(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer("label_to_bio", torch.tensor(LABEL_TO_BIO))
        if config.focal_alpha is not None and len(config.focal_alpha) != len(LABELS):
            raise ValueError("focal_alpha must have 19 entries in LABELS order")

    def auxiliary(self, output, labels, mask):
        return F.cross_entropy(
            output.bio_logits[mask].float(),
            self.label_to_bio[labels[mask]],
        )

    def stage1(self, student, output, teacher_output, labels, mask):
        auxiliary = self.auxiliary(output, labels, mask)
        mse = masked_state_mse(student.state_projection(output.hidden), teacher_output.hidden, mask)
        kd = masked_distillation(
            output.logits,
            teacher_output.logits,
            mask,
            self.config.temperature,
        )
        total = (
            self.config.mse_weight * mse
            + self.config.kd_weight * kd
            + self.config.auxiliary_weight * auxiliary
        )
        return {"loss": total, "mse": mse, "kd": kd, "auxiliary": auxiliary}

    def stage2(self, student, output, labels, mask):
        auxiliary = self.auxiliary(output, labels, mask)
        crf = student.crf_loss(output.logits, labels, mask)
        focal = masked_focal_loss(
            output.logits,
            labels,
            mask,
            self.config.focal_gamma,
            self.config.focal_alpha,
        )
        total = crf + self.config.focal_weight * focal + self.config.auxiliary_weight * auxiliary
        return {"loss": total, "crf": crf, "focal": focal, "auxiliary": auxiliary}

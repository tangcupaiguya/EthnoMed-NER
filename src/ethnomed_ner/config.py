import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BADPConfig:
    student_backbone: str | None = None
    teacher_backbone: str | None = None
    tokenizer: str | None = None
    backbone_dim: int = 768
    teacher_dim: int = 768
    model_dim: int = 256
    boundary_dim: int = 32
    # Manuscript complex dimension; not an upstream real-storage d_state argument.
    complex_state_dim: int = 16
    mimo_rank: int = 4
    classifier_dropout: float = 0.2
    backbone_dropout: float = 0.1
    temperature: float = 3.0
    mse_weight: float = 1.0
    kd_weight: float = 1.0
    auxiliary_weight: float = 0.5
    focal_weight: float = 1.0
    focal_gamma: float = 2.0
    focal_alpha: list[float] | None = None
    window_size: int = 512
    window_stride: int = 256
    teacher_epochs: int = 10
    stage1_epochs: int = 8
    stage2_epochs: int = 22
    teacher_lr: float = 3e-5
    stage1_lr: float = 2e-5
    stage2_lr: float = 1e-5
    batch_size: int = 16
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    patience: int = 5
    seed: int = 42
    amp: bool = True
    lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    def __post_init__(self):
        if self.temperature <= 0 or self.focal_gamma < 0:
            raise ValueError("temperature must be positive and focal_gamma nonnegative")
        if not 0 < self.window_stride <= self.window_size - 2:
            raise ValueError("window_stride must fit within the content window")
        if min(self.model_dim, self.complex_state_dim, self.mimo_rank) < 1:
            raise ValueError("Model and state dimensions and MIMO rank must be positive")
        if min(self.teacher_epochs, self.stage1_epochs, self.stage2_epochs, self.batch_size) < 1:
            raise ValueError("Epoch counts and batch_size must be positive")
        if not 0 <= self.warmup_ratio <= 1 or self.patience < 0:
            raise ValueError("Invalid warmup_ratio or patience")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

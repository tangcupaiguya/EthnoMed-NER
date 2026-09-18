# EthnoMed-NER: BADP Core Modules

Core implementation accompanying **BADP: Ethnomedicine Named Entity Recognition via Boundary-Aware Mamba and Decoupled Procedural Distillation** (`ASOC-D-26-07132_R1.pdf`).

This release contains the boundary-prior components, two-stage objectives, parameter-freezing schedule, CRF and token-alignment utilities described in the revised manuscript. It also provides a bidirectional encoder interface. See [the manuscript-to-code mapping](docs/method_mapping.md) for details.

## Architecture

```text
MC-BERT hidden states (768)
  -> bottleneck (768 -> 256)
  -> predicted soft BIO distribution (256 -> 3)
  -> learned boundary feature (3 x 32)
  -> concatenate token and boundary features (256 + 32)
  -> input projection (288 -> 256)
  -> independent forward / backward Mamba-3 MIMO (256 each)
  -> concatenate directional states (512)
  -> dropout (0.2) + classifier (512 -> 19)
  -> linear-chain CRF
```

Gold BIO labels supervise the auxiliary loss; they never enter the boundary-prior forward path. The teacher is a RoBERTa encoder with a linear `768 -> 19` head. It is used only during Stage 1 of student training.

## Files and manuscript correspondence

| File | Core implementation | Manuscript reference |
| --- | --- | --- |
| [boundary.py](src/ethnomed_ner/boundary.py) | Bottleneck, differentiable soft BIO prior, boundary embedding and fusion | Section 3.2.2, Eqs. (4)-(5) |
| [encoder.py](src/ethnomed_ner/encoder.py) | Bidirectional interface, RMSNorm, residuals and padding-safe reversal; Mamba-3 backend supplied externally | Section 3.2, Eq. (6), Table 6 |
| [model.py](src/ethnomed_ner/model.py) | BADP student, RoBERTa teacher, CRF, state alignment and stage freezing | Eqs. (6), (8), (24), Algorithm 1 |
| [losses.py](src/ethnomed_ner/losses.py) | Masked MSE, temperature-scaled KL, auxiliary BIO CE, CRF/focal objective | Eqs. (9)-(12) |
| [data.py](src/ethnomed_ner/data.py) | Flat-span projection, character-to-token labels, shared windows and mean pooling | Appendix A, Section 4.8 |
| [labels.py](src/ethnomed_ner/labels.py) | Nine entity types, 19 BIO labels and three auxiliary BIO labels | Section 3.1, Appendix A |
| [config.py](src/ethnomed_ner/config.py), [badp.json](configs/badp.json) | Architecture, loss and schedule settings; optional attention-only LoRA | Tables 5-6, Appendix E |

## Two-stage objectives

For valid content-token positions only:

```text
Stage 1 = MSE(projected student states, teacher states)
        + tau^2 * KL(teacher probabilities || student probabilities)
        + 0.5 * auxiliary BIO cross-entropy

Stage 2 = CRF negative log-likelihood
        + 1.0 * focal loss (gamma = 2)
        + 0.5 * auxiliary BIO cross-entropy
```

The default temperature is `tau = 3`. The temperature-squared factor appears exactly once. MSE averages valid tokens and hidden dimensions; KL sums labels and averages valid tokens. Focal loss and auxiliary CE average valid tokens. Padding and inserted special tokens are excluded.

| Phase | Epochs / learning rate | Parameter policy |
| --- | --- | --- |
| Teacher fine-tuning | 10 / `3e-5` | RoBERTa + linear head; masked token CE |
| Stage 1 | 8 / `2e-5` | Freeze student token lookup embeddings and unused CRF; train Transformer layers, added modules and state projection; teacher frozen/eval |
| Stage 2 | 22 / `1e-5` | Unfreeze student token lookup embeddings; freeze state projection; train CRF; remove both teacher losses |

For the optional LoRA variant, the original backbone stays frozen in both stages. Attention query/key/value/output adapters use rank 16, alpha 32 and dropout 0.05.

## Installation

Use Python 3.10+:

```bash
python -m pip install -e .
# Optional LoRA support:
python -m pip install -e ".[lora]"
```

Full student execution additionally requires the original Mamba-3 MIMO implementation and a compatible GPU environment. Pass a `mixer_factory(config)` that constructs a fresh causal module for each direction, mapping `[batch, length, 256]` to the same shape. The wrapper supplies RMSNorm and the outer residual. Without a factory, construction raises an explicit error.

## Core API

Supply the exact MC-BERT, RoBERTa and shared-tokenizer checkpoints used in your experiment. A shared vocabulary, token-to-ID mapping and special-token IDs are required for teacher/student state alignment.

```python
import torch

from ethnomed_ner.config import BADPConfig
from ethnomed_ner.losses import BADPObjective
from ethnomed_ner.model import BADP, NERTeacher, load_backbone

config = BADPConfig.load("configs/badp.json")
# Set config.student_backbone and config.teacher_backbone to your checkpoints.
# original_mamba3_factory must use your verified backend configuration.
student = BADP(load_backbone(config.student_backbone), config, original_mamba3_factory)
teacher = NERTeacher(load_backbone(config.teacher_backbone), config.teacher_dim)
# Load the task-fine-tuned teacher state before freezing it.
teacher.freeze()
objective = BADPObjective(config)

# batch comes from WindowCollator; mask covers content tokens only.
student.set_stage(1)
with torch.no_grad():
    teacher_output = teacher(batch)
output = student(batch)
losses = objective.stage1(student, output, teacher_output, batch["labels"], batch["mask"])

# After Stage 1, rebuild the optimizer over trainable student parameters.
student.set_stage(2)
output = student(batch)
losses = objective.stage2(student, output, batch["labels"], batch["mask"])
predicted_tags = student.decode(output, batch["mask"])
```

This illustrates the API and objective switch; it is not a complete training script. Put the models, objective and batch on the same device, and use FP16 autocast with gradient scaling for the Mamba-3 training configuration.

`EthnoMedDataset` accepts UTF-8 JSONL records with `id`, `text`, and `entities`, where each entity has `start`, `end`, and `label`. Offsets are zero-based Unicode character positions with an exclusive end. `WindowCollator` adds special tokens and creates overlapping backbone windows; `move_batch` moves their tensors to a device.

## Scope and verification

This core release provides no corpus, checkpoints, experiment results, training CLI or evaluation scripts. Surrounding core behavior was verified locally on CPU with small BERT models and an explicitly identified causal test double. Real CUDA Mamba-3 kernels and the reported EthnoMed scores have not been validated by this release.

The configuration records the manuscript's `complex_state_dim=16` and `mimo_rank=4`; it makes no automatic conversion to an upstream `d_state`. Backend construction is deferred until the original parameters are confirmed. Exact pretrained-model identifiers and focal alpha values also require the original settings; see [method_mapping.md](docs/method_mapping.md).

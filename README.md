# EthnoMed-NER / BADP

Core modules for **BADP: Ethnomedicine Named Entity Recognition via Boundary-Aware Mamba and Decoupled Procedural Distillation**.

## Core modules

| Module | Implementation | Paper |
| --- | --- | --- |
| [boundary.py](src/ethnomed_ner/boundary.py) | Bottleneck, soft BIO boundary prior and feature fusion | Eqs. (4)-(5) |
| [encoder.py](src/ethnomed_ner/encoder.py) | Bidirectional encoder wrapper, RMSNorm and residual connections | Eq. (6), Table 6 |
| [model.py](src/ethnomed_ner/model.py) | Student/teacher models, state alignment, CRF, stage freezing and entity prediction | Eqs. (6), (8), (24), Algorithm 1 |
| [losses.py](src/ethnomed_ner/losses.py) | Masked MSE, KL distillation, auxiliary BIO loss and focal loss | Eqs. (9)-(12) |
| [data.py](src/ethnomed_ner/data.py) | Span projection, character offsets and batched overlapping-window pooling | Appendix A, Section 4.8 |
| [labels.py](src/ethnomed_ner/labels.py) | Nine entity types and 19 BIO labels | Section 3.1 |
| [config.py](src/ethnomed_ner/config.py), [badp.json](configs/badp.json) | Architecture and loss parameters | Tables 5-6 |

## Model

```text
MC-BERT (768) -> Bottleneck (256) + Soft BIO Prior (32)
             -> Fusion (256) -> Bidirectional Mamba-3 (512)
             -> Dropout + Linear (19) -> CRF
```

Stage 1 combines hidden-state MSE, logit KL at temperature 3 and auxiliary BIO supervision. Stage 2 combines CRF loss, focal loss (gamma 2) and auxiliary BIO supervision. The auxiliary loss weight is 0.5 in both stages.

## Usage

```bash
python -m pip install -e .
```

```python
from ethnomed_ner import BADP, BADPConfig, BADPObjective

config = BADPConfig.load("configs/badp.json")
# backbone: MC-BERT; mixer_factory(config): a fresh Mamba-3 MIMO block.
student = BADP(backbone, config, mixer_factory=mixer_factory)
objective = BADPObjective(config)
student.set_stage(1)  # Procedural priming; build the optimizer after selecting a stage.
# objective.stage1(student, output, teacher_output, labels, mask)
student.set_stage(2)  # Decoupled specialization; rebuild the optimizer for this stage.
# objective.stage2(student, output, labels, mask)

entities = student.predict(batch, window_batch_size=8)
# Each document returns Span(start, end, label), with exclusive character end offsets.
```

`predict` runs CRF decoding in evaluation mode and restores the prior model mode. Window batches are pooled into full-document states before bidirectional encoding. A frozen teacher stays in evaluation mode during distillation.

## Engineering references

[Transformers](https://github.com/huggingface/transformers) for character-offset entity outputs, [Flair](https://github.com/flairNLP/flair) for NER prediction interfaces, and [Mamba](https://github.com/state-spaces/mamba) for modular sequence mixers and RMSNorm.

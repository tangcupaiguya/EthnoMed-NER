import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .labels import ENTITY_TYPES, LABEL_TO_ID


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int
    label: str


def project_flat_spans(spans, text_length):
    """Appendix A: deduplicate, discard strict inner spans, then resolve partial overlap."""
    unique = sorted(set(spans))
    boundaries = {}
    for span in unique:
        if not 0 <= span.start < span.end <= text_length or span.label not in ENTITY_TYPES:
            raise ValueError(f"Invalid span: {span}")
        key = (span.start, span.end)
        if key in boundaries and boundaries[key] != span.label:
            raise ValueError("Same-boundary type conflicts require expert adjudication")
        boundaries[key] = span.label
    outer = [
        span
        for span in unique
        if not any(
            other != span and other.start <= span.start and span.end <= other.end
            for other in unique
        )
    ]
    priority = {kind: index for index, kind in enumerate(ENTITY_TYPES)}
    retained = []
    for span in sorted(outer, key=lambda item: (priority[item.label], item.start, -item.end)):
        if all(span.end <= other.start or other.end <= span.start for other in retained):
            retained.append(span)
    return sorted(retained)


def align_span_labels(offsets, spans):
    labels = [LABEL_TO_ID["O"]] * len(offsets)
    for span in spans:
        positions = [
            i for i, (start, end) in enumerate(offsets) if start < span.end and span.start < end
        ]
        if (
            not positions
            or offsets[positions[0]][0] != span.start
            or offsets[positions[-1]][1] != span.end
        ):
            raise ValueError(f"Entity boundary falls inside or outside tokenizer tokens: {span}")
        for position in positions:
            prefix = "B" if position == positions[0] else "I"
            labels[position] = LABEL_TO_ID[f"{prefix}-{span.label}"]
    return labels


class EthnoMedDataset(Dataset):
    """JSONL with id, text, and adjudicated character-offset entities (end exclusive)."""

    def __init__(self, path, tokenizer, labeled=True):
        if not tokenizer.is_fast:
            raise ValueError("A fast shared tokenizer is required for character offsets")
        self.records = []
        seen = set()
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                doc_id = str(record.get("id", line_number))
                if doc_id in seen:
                    raise ValueError(f"Duplicate document id: {doc_id}")
                seen.add(doc_id)
                text = record["text"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Document {doc_id} has empty or invalid text")
                if labeled and "entities" not in record:
                    raise ValueError(f"Document {doc_id} is missing entities")
                spans = project_flat_spans(
                    [Span(**entity) for entity in record.get("entities", [])],
                    len(text),
                )
                encoded = tokenizer(
                    text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                    truncation=False,
                    verbose=False,
                )
                ids = encoded["input_ids"]
                offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
                if not ids or any(start >= end for start, end in offsets):
                    raise ValueError(f"Document {doc_id} has empty or zero-width token offsets")
                labels = align_span_labels(offsets, spans)
                self.records.append(
                    {
                        "id": doc_id,
                        "text": text,
                        "spans": spans,
                        "input_ids": ids,
                        "offsets": offsets,
                        "labels": labels,
                    }
                )
        if not self.records:
            raise ValueError(f"Empty dataset: {path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class WindowCollator:
    """512 total positions, stride 256; merge repeated content states before the upper encoder."""

    def __init__(self, tokenizer, window_size=512, stride=256):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.capacity = window_size - tokenizer.num_special_tokens_to_add(pair=False)
        self.stride = stride
        if not 0 < stride <= self.capacity:
            raise ValueError("Window stride leaves uncovered content positions")
        if tokenizer.padding_side != "right":
            raise ValueError("Use right padding for shared encoder windows")

    def __call__(self, records):
        max_length = max(len(record["input_ids"]) for record in records)
        labels = torch.full((len(records), max_length), -100, dtype=torch.long)
        mask = torch.zeros_like(labels, dtype=torch.bool)
        windows, mappings = [], []
        for row, record in enumerate(records):
            ids = record["input_ids"]
            labels[row, : len(ids)] = torch.tensor(record["labels"])
            mask[row, : len(ids)] = True
            start = 0
            while True:
                end = min(start + self.capacity, len(ids))
                content = ids[start:end]
                token_ids = self.tokenizer.build_inputs_with_special_tokens(content)
                # A sentinel identifies inserted specials without confusing literal [UNK]/[SEP] text.
                marker = self.tokenizer.build_inputs_with_special_tokens([-1] * len(content))
                content_positions = [i for i, value in enumerate(marker) if value == -1]
                if len(content_positions) != len(content):
                    raise ValueError("Tokenizer changed content positions while adding specials")
                item = {"input_ids": token_ids, "attention_mask": [1] * len(token_ids)}
                if "token_type_ids" in self.tokenizer.model_input_names:
                    item["token_type_ids"] = self.tokenizer.create_token_type_ids_from_sequences(
                        content
                    )
                mapping = [-1] * len(token_ids)
                for position, original in zip(content_positions, range(start, end)):
                    mapping[position] = row * max_length + original
                windows.append(item)
                mappings.append(mapping)
                if end == len(ids):
                    break
                start += self.stride
        encoded = dict(self.tokenizer.pad(windows, padding=True, return_tensors="pt"))
        width = encoded["input_ids"].shape[1]
        mapping = torch.tensor([item + [-1] * (width - len(item)) for item in mappings])
        return {
            "windows": encoded,
            "window_mapping": mapping,
            "mask": mask,
            "labels": labels,
            "records": records,
        }


def move_batch(batch, device):
    return {
        **batch,
        "windows": {key: value.to(device) for key, value in batch["windows"].items()},
        **{key: batch[key].to(device) for key in ("window_mapping", "mask", "labels")},
    }


def encode_windows(backbone, batch):
    hidden = backbone(**batch["windows"]).last_hidden_state
    mapping = batch["window_mapping"]
    valid = mapping >= 0
    batch_size, length = batch["mask"].shape
    # Float32 accumulation avoids low-precision errors where overlapping windows are averaged.
    merged = hidden.new_zeros((batch_size * length, hidden.shape[-1]), dtype=torch.float32)
    merged = merged.index_add(0, mapping[valid], hidden[valid].float())
    counts = hidden.new_zeros(batch_size * length, dtype=torch.float32)
    counts.index_add_(0, mapping[valid], torch.ones_like(mapping[valid], dtype=torch.float32))
    if (counts.view(batch_size, length)[batch["mask"]] == 0).any():
        raise ValueError("Uncovered content token in the window mapping")
    return (
        (merged / counts.clamp_min(1).unsqueeze(-1)).to(hidden.dtype).view(batch_size, length, -1)
    )

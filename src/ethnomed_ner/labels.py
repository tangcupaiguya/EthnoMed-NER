"""Appendix A category precedence and a stable 19-label BIO inventory."""

ENTITY_TYPES = (
    "Symptom",
    "Function",
    "Name",
    "Alias",
    "Species",
    "Part",
    "Original",
    "Pharmacopoeia",
    "Family",
)
LABELS = ("O",) + tuple(f"{prefix}-{kind}" for kind in ENTITY_TYPES for prefix in ("B", "I"))
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
BIO_TO_ID = {"B": 0, "I": 1, "O": 2}
LABEL_TO_BIO = tuple(BIO_TO_ID[label.split("-")[0]] for label in LABELS)

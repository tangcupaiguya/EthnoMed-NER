"""Core modules for BADP, reconstructed from the revised manuscript."""

from .config import BADPConfig
from .labels import ENTITY_TYPES, LABELS

__all__ = ["ENTITY_TYPES", "LABELS", "BADPConfig"]

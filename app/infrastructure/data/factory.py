"""Data loader factory -- pick the right loader based on file extension.

Adding a new format (Excel, Parquet, JSON-lines, ...) only requires:
1. Implementing DataLoader in a new file.
2. Adding one entry to the registry below.
"""
from pathlib import Path

from .base import DataLoader
from .csv_loader import CSVLoader


# Registry: extension -> loader class. Multiple extensions can map to the same
# loader (e.g. CSVLoader handles .csv, .tsv, .txt).
_LOADERS = [
    CSVLoader,
]


def get_loader_for_file(file_path: str) -> DataLoader:
    """Return a loader instance that can handle the given file."""
    ext = Path(file_path).suffix.lower()
    for loader_cls in _LOADERS:
        loader = loader_cls()
        if ext in loader.supported_extensions():
            return loader
    raise ValueError(f"No loader available for file extension: {ext}")

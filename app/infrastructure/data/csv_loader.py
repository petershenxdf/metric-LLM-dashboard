"""CSV loader.

Auto-detects the delimiter, drops non-numeric columns (with a warning), and
returns a clean numeric DataFrame indexed 0..n-1. The dashboard uses the row
index as the canonical point_id throughout, so we reset_index() at the end.
"""
from typing import List, Tuple, Optional
import csv
import numpy as np
import pandas as pd

from .base import DataLoader


class CSVLoader(DataLoader):
    def supported_extensions(self) -> List[str]:
        return [".csv", ".tsv", ".txt"]

    def load(self, file_path: str) -> pd.DataFrame:
        """Return only the normalized DataFrame (for backward compatibility)."""
        _, normalized = self.load_both(file_path)
        return normalized

    def load_both(self, file_path: str) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
        """Return both raw (non-normalized) and z-score normalized DataFrames.

        The raw DataFrame keeps all original numeric columns in their original
        scale -- this is what gets written back out when the user exports their
        labeled dataset to CSV. The normalized DataFrame is what the clustering
        pipeline actually consumes.
        """
        delimiter = self._detect_delimiter(file_path)
        df = pd.read_csv(file_path, sep=delimiter)

        # Drop fully empty rows / columns
        df = df.dropna(how="all").dropna(axis=1, how="all")

        # Keep only numeric columns. Non-numeric ones become a warning in
        # validate().
        numeric_df = df.select_dtypes(include=[np.number])

        # Fill remaining NaNs with column mean
        numeric_df = numeric_df.fillna(numeric_df.mean(numeric_only=True))

        # Raw version: numeric columns in original units, indexed 0..n-1
        raw = numeric_df.reset_index(drop=True)

        # Normalized version: z-score per column, constant columns -> zero
        means = raw.mean()
        stds = raw.std().replace(0, 1.0)
        normalized = ((raw - means) / stds).reset_index(drop=True)

        return raw, normalized

    def validate(self, df: pd.DataFrame) -> List[str]:
        warnings = []
        if len(df) == 0:
            warnings.append("Loaded file is empty")
        if df.shape[1] == 0:
            warnings.append("No numeric columns found")
        if len(df) > 5000:
            warnings.append(
                f"Dataset has {len(df)} points -- clustering may be slow. "
                "Consider sub-sampling for interactive use."
            )
        if df.shape[1] > 100:
            warnings.append(
                f"Dataset has {df.shape[1]} features -- consider PCA "
                "preprocessing for better metric learning behavior."
            )
        return warnings

    @staticmethod
    def _detect_delimiter(file_path: str) -> str:
        """Sniff the delimiter from the first few KB of the file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            return dialect.delimiter
        except csv.Error:
            return ","

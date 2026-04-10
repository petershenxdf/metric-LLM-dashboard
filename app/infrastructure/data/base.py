"""Abstract base class for data loaders."""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import pandas as pd


class DataLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> pd.DataFrame:
        """Load a file and return a clean numeric DataFrame ready for clustering."""
        ...

    def load_both(self, file_path: str) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
        """Load both raw and cleaned/normalized versions of the file.

        Default implementation returns (None, load(file_path)) so subclasses
        that do not distinguish raw vs cleaned still work. Subclasses that
        normalize can override to return the pre-normalized DataFrame as the
        first element -- useful for CSV export.
        """
        return None, self.load(file_path)

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> List[str]:
        """Inspect the loaded DataFrame and return a list of warnings (may be empty)."""
        ...

    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return file extensions this loader handles, e.g. ['.csv']."""
        ...

"""Base adapter interface for model outputs."""

from abc import ABC, abstractmethod
from benchmark.schema import ImageAnnotation


class ModelAdapter(ABC):
    """Abstract base for model adapters.

    Each adapter normalizes a specific model's output format into the
    common ImageAnnotation schema used by the evaluator.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model name for reporting."""
        ...

    @abstractmethod
    def parse(self, raw_output, image_id: str, image_path: str) -> ImageAnnotation:
        """Parse raw model output into ImageAnnotation.

        Args:
            raw_output: Model output in its native format (dict, str, etc.).
            image_id: Image identifier.
            image_path: Path to the source image file.

        Returns:
            Normalized ImageAnnotation.
        """
        ...
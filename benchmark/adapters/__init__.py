"""Model adapters for normalizing diverse model outputs."""

from benchmark.adapters.base import ModelAdapter
from benchmark.adapters.paddle_ocr import PaddleOCRAdapter, PaddleOCRVAdapter
from benchmark.adapters.gemini import GeminiAdapter

__all__ = ["ModelAdapter", "PaddleOCRAdapter", "PaddleOCRVAdapter", "GeminiAdapter"]
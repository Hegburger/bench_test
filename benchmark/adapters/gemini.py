"""Adapter for Gemini multimodal model output.

Gemini is prompted to produce a structured JSON description of the document
page, including region bounding boxes, labels, and text transcriptions.

Expected output format:
{
  "regions": [
    {
      "bbox": [x1, y1, x2, y2],  // or "points": [[x1,y1], ...]
      "label": "text" | "liding" | "oracle" | "picture",
      "transcription": "..."
    }
  ]
}
"""

import json
import re

from benchmark.adapters.base import ModelAdapter
from benchmark.schema import ImageAnnotation, Region


class GeminiAdapter(ModelAdapter):
    """Parse Gemini JSON/markdown output into ImageAnnotation."""

    @property
    def model_name(self) -> str:
        return "Gemini"

    def parse(self, raw_output, image_id: str, image_path: str) -> ImageAnnotation:
        """Parse Gemini output, handling JSON-in-markdown and plain JSON."""
        if isinstance(raw_output, str):
            data = self._extract_json(raw_output)
        elif isinstance(raw_output, dict):
            data = raw_output
        else:
            data = {}

        regions = []
        raw_regions = data.get("regions", [])
        if not raw_regions and isinstance(data, list):
            raw_regions = data

        for item in raw_regions:
            # Handle bbox formats
            bbox = None
            if "bbox" in item:
                bbox = item["bbox"]
            elif "points" in item:
                pts = item["points"]
                if len(pts) == 4:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]

            if bbox is None or len(bbox) != 4:
                continue

            region = Region(
                bbox=[float(v) for v in bbox],
                label=item.get("label", "text"),
                transcription=item.get("transcription", ""),
                confidence=float(item.get("confidence", 1.0)),
            )
            regions.append(region)

        return ImageAnnotation(
            image_id=image_id,
            image_path=image_path,
            regions=regions,
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from markdown code blocks or raw text."""
        # Try markdown ```json block first
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # Try raw JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find a JSON object in the text
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        return {}
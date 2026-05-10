"""Adapter for PaddleOCR / PaddleOCR-VL output formats."""

import json
from pathlib import Path

from benchmark.adapters.base import ModelAdapter
from benchmark.schema import ImageAnnotation, Region

# PaddleOCR-VL layout labels → benchmark label types
_VL_LABEL_MAP = {
    "text": "text",
    "vertical_text": "text",
    "number": "text",
    "header": "text",
    "footer": "text",
    "footnote": "text",
    "aside_text": "text",
    "formula": "text",
    "header_image": "picture",
    "footer_image": "picture",
    "image": "picture",
    "figure": "picture",
    "table": "picture",
    "chart": "picture",
    "seal": "picture",
}


class PaddleOCRAdapter(ModelAdapter):
    """Parse PaddleOCR JSON output into ImageAnnotation."""

    @property
    def model_name(self) -> str:
        return "PaddleOCR"

    def parse(self, raw_output, image_id: str, image_path: str) -> ImageAnnotation:
        if isinstance(raw_output, str):
            data = json.loads(raw_output)
        else:
            data = raw_output

        regions = []
        for item in data:
            points = item.get("points", item.get("bbox", []))
            text = item.get("text", item.get("transcription", ""))
            confidence = item.get("confidence", item.get("score", 1.0))

            if len(points) != 4:
                continue

            region = Region.from_points(
                points=points,
                label="text",
                transcription=text,
                confidence=float(confidence),
            )
            regions.append(region)

        return ImageAnnotation(
            image_id=image_id,
            image_path=image_path,
            regions=regions,
        )


class PaddleOCRVAdapter(ModelAdapter):
    """Parse PaddleOCR-VL (PP-OCR-VL) JSON output into ImageAnnotation.

    Handles the nested prunedResult.parsing_res_list format.
    """

    @property
    def model_name(self) -> str:
        return "PaddleOCR-VL-1.5"

    def parse(self, raw_output, image_id: str, image_path: str) -> ImageAnnotation:
        if isinstance(raw_output, str):
            data = json.loads(raw_output)
        else:
            data = raw_output

        parsing_list = self._extract_parsing_list(data)
        if parsing_list is None:
            return ImageAnnotation(image_id=image_id, image_path=image_path, regions=[])

        regions = []
        for block in parsing_list:
            block_label = block.get("block_label", "text")
            content = block.get("block_content", "")
            bbox = block.get("block_bbox", [])

            if len(bbox) != 4:
                continue

            label = _VL_LABEL_MAP.get(block_label, "text")
            region = Region(
                bbox=[float(v) for v in bbox],
                label=label,
                transcription=content,
                confidence=1.0,
            )
            regions.append(region)

        return ImageAnnotation(
            image_id=image_id,
            image_path=image_path,
            regions=regions,
        )

    @staticmethod
    def _extract_parsing_list(data) -> list | None:
        """Extract parsing_res_list from various PaddleOCR-VL formats."""
        if isinstance(data, list) and len(data) > 0:
            item = data[0]
        elif isinstance(data, dict):
            item = data
        else:
            return None

        pruned = item.get("prunedResult", item)
        return pruned.get("parsing_res_list", None)
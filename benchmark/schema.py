"""Core data structures for the benchmark."""

from dataclasses import dataclass, field


@dataclass
class Region:
    """A detected/annotated region on a document image."""

    bbox: list[float]  # [x1, y1, x2, y2] in absolute pixels
    label: str  # "text", "liding", "oracle", "picture"
    transcription: str = ""  # text content or IDS string; empty for oracle/picture
    confidence: float = 1.0  # model confidence (always 1.0 for GT)

    @property
    def points(self) -> list[list[float]]:
        """Return four corner points [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ]."""
        x1, y1, x2, y2 = self.bbox
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, (x2 - x1) * (y2 - y1))

    @classmethod
    def from_points(cls, points: list[list[float]], label: str,
                    transcription: str = "", confidence: float = 1.0) -> "Region":
        """Create a Region from four corner points."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return cls(
            bbox=[min(xs), min(ys), max(xs), max(ys)],
            label=label,
            transcription=transcription,
            confidence=confidence,
        )


@dataclass
class ImageAnnotation:
    """Complete annotation for a single image."""

    image_id: str
    image_path: str
    regions: list[Region] = field(default_factory=list)

    @property
    def region_count(self) -> int:
        return len(self.regions)

    def regions_by_label(self, label: str) -> list[Region]:
        return [r for r in self.regions if r.label == label]

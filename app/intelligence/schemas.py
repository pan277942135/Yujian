from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConfusionPair:
    """A single directed true-species -> predicted-species error pair."""

    true_species: str
    pred_species: str
    error_count: int
    error_rate: float
    priority: str
    priority_score: float = 0.0
    species_importance: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ``dict``/``model_dump`` make this small dataclass convenient for callers
    # that already use Pydantic-style serialization elsewhere in the Console.
    def dict(self) -> dict[str, Any]:
        return self.to_dict()

    def model_dump(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass
class ConfusionReport:
    model_version: str
    generated_at: str
    top_confusions: list[ConfusionPair] = field(default_factory=list)
    source: str | None = None
    total_samples: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["top_confusions"] = [pair.to_dict() for pair in self.top_confusions]
        return result

    def dict(self) -> dict[str, Any]:
        return self.to_dict()

    def model_dump(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(frozen=True)
class DataGap:
    species: str
    current: int
    target: int
    gap: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CollectionSpeciesRequirement:
    name: str
    count: int
    target: int = 0
    gap: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CollectionTask:
    task_id: str
    task_type: str
    model_version: str
    generated_at: str
    status: str
    reason: list[dict[str, Any]] = field(default_factory=list)
    requirements: dict[str, Any] = field(default_factory=dict)
    batch_suggestion: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dict(self) -> dict[str, Any]:
        return self.to_dict()

    def model_dump(self) -> dict[str, Any]:
        return self.to_dict()

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class StageDefinition:
    stage_id: str
    estimated_resident_bytes: int
    dependencies: List[str] = field(default_factory=list)
    entrypoints: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterGraph:
    architecture_id: str
    architecture_version: str
    stages: List[StageDefinition]
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ShardPlan:
    shard_id: str
    stage_ids: List[str]
    estimated_resident_bytes: int
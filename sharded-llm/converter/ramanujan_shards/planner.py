from typing import Dict, List, Set

from .contracts import AdapterGraph, ShardPlan, StageDefinition


def plan_shards(graph: AdapterGraph, requested_shards: int) -> List[ShardPlan]:
    if requested_shards <= 0:
        raise ValueError("requested_shards must be positive")
    if not graph.stages:
        raise ValueError("adapter graph must contain at least one stage")

    ordered = _topological_order(graph.stages)
    shard_count = min(requested_shards, len(ordered))
    remaining_bytes = sum(stage.estimated_resident_bytes for stage in ordered)
    plans: List[ShardPlan] = []
    current: List[StageDefinition] = []
    current_bytes = 0

    for index, stage in enumerate(ordered):
        stages_left = len(ordered) - index
        shards_left = shard_count - len(plans)
        target_bytes = max(1, remaining_bytes // shards_left)
        should_close = (
            current
            and current_bytes + stage.estimated_resident_bytes > target_bytes
            and stages_left >= shards_left
        )
        if should_close:
            plans.append(_make_plan(len(plans), current, current_bytes))
            remaining_bytes -= current_bytes
            current = []
            current_bytes = 0

        current.append(stage)
        current_bytes += stage.estimated_resident_bytes

        stages_after = len(ordered) - index - 1
        shards_after = shard_count - len(plans) - 1
        if shards_after > 0 and stages_after == shards_after:
            plans.append(_make_plan(len(plans), current, current_bytes))
            remaining_bytes -= current_bytes
            current = []
            current_bytes = 0

    if current:
        plans.append(_make_plan(len(plans), current, current_bytes))
    return plans


def _make_plan(index: int, stages: List[StageDefinition], resident_bytes: int) -> ShardPlan:
    return ShardPlan(
        shard_id="shard-{0:02d}".format(index),
        stage_ids=[stage.stage_id for stage in stages],
        estimated_resident_bytes=resident_bytes,
    )


def _topological_order(stages: List[StageDefinition]) -> List[StageDefinition]:
    by_id: Dict[str, StageDefinition] = {}
    declaration_order: Dict[str, int] = {}
    for index, stage in enumerate(stages):
        if not stage.stage_id:
            raise ValueError("every stage requires stage_id")
        if stage.stage_id in by_id:
            raise ValueError("duplicate stage_id: {0}".format(stage.stage_id))
        if stage.estimated_resident_bytes < 0:
            raise ValueError("stage {0} has negative resident bytes".format(stage.stage_id))
        by_id[stage.stage_id] = stage
        declaration_order[stage.stage_id] = index

    dependents: Dict[str, List[str]] = {stage_id: [] for stage_id in by_id}
    indegree: Dict[str, int] = {stage_id: 0 for stage_id in by_id}
    for stage in stages:
        seen_dependencies: Set[str] = set()
        for dependency in stage.dependencies:
            if dependency not in by_id:
                raise ValueError(
                    "stage {0} depends on unknown stage: {1}".format(stage.stage_id, dependency)
                )
            if dependency in seen_dependencies:
                continue
            seen_dependencies.add(dependency)
            dependents[dependency].append(stage.stage_id)
            indegree[stage.stage_id] += 1

    ready = [stage_id for stage_id, degree in indegree.items() if degree == 0]
    ready.sort(key=declaration_order.get)
    ordered: List[StageDefinition] = []
    while ready:
        stage_id = ready.pop(0)
        ordered.append(by_id[stage_id])
        for dependent in dependents[stage_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
                ready.sort(key=declaration_order.get)

    if len(ordered) != len(stages):
        raise ValueError("adapter graph contains a cycle")
    return ordered
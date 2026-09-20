package in.ramanujan.shardedllm.model;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class ManifestValidator {
    private ManifestValidator() {
    }

    public static List<String> validate(ModelPackageManifest manifest) {
        List<String> errors = new ArrayList<>();
        if (manifest == null) {
            errors.add("manifest is required");
            return errors;
        }

        require(manifest.getSchemaVersion(), "schemaVersion", errors);
        require(manifest.getPackageId(), "packageId", errors);
        require(manifest.getArchitectureId(), "architectureId", errors);

        Map<String, ExecutionShardManifest> shards = new HashMap<>();
        for (ExecutionShardManifest shard : safe(manifest.getShards())) {
            if (shard == null || blank(shard.getShardId())) {
                errors.add("every shard requires shardId");
                continue;
            }
            if (shards.put(shard.getShardId(), shard) != null) {
                errors.add("duplicate shardId: " + shard.getShardId());
            }
            validateRelativePath(shard.getIrPath(), "shard " + shard.getShardId() + " irPath", errors);
            validateShard(shard, errors);
        }
        if (shards.isEmpty()) {
            errors.add("at least one shard is required");
        }

        Map<String, ExecutionNode> nodes = new HashMap<>();
        for (ExecutionNode node : safe(manifest.getExecutionGraph())) {
            if (node == null || blank(node.getNodeId())) {
                errors.add("every execution node requires nodeId");
                continue;
            }
            if (nodes.put(node.getNodeId(), node) != null) {
                errors.add("duplicate nodeId: " + node.getNodeId());
            }
            ExecutionShardManifest shard = shards.get(node.getShardId());
            if (shard == null) {
                errors.add("node " + node.getNodeId() + " references unknown shard: " + node.getShardId());
            } else if (!hasEntrypoint(shard, node.getEntrypoint())) {
                errors.add("node " + node.getNodeId() + " references unknown entrypoint: " + node.getEntrypoint());
            }
        }
        for (ExecutionNode node : nodes.values()) {
            for (String dependency : safe(node.getDependsOn())) {
                if (!nodes.containsKey(dependency)) {
                    errors.add("node " + node.getNodeId() + " depends on unknown node: " + dependency);
                }
            }
        }
        detectCycles(nodes, errors);
        return errors;
    }

    public static void validateOrThrow(ModelPackageManifest manifest) {
        List<String> errors = validate(manifest);
        if (!errors.isEmpty()) {
            throw new IllegalArgumentException(String.join("; ", errors));
        }
    }

    private static void validateShard(ExecutionShardManifest shard, List<String> errors) {
        Set<String> entrypointNames = new HashSet<>();
        Set<String> stateNames = new HashSet<>();
        for (StateContract state : safe(shard.getStates())) {
            if (state == null || state.getTensor() == null || blank(state.getTensor().getName())) {
                errors.add("shard " + shard.getShardId() + " has state without a tensor name");
            } else if (!stateNames.add(state.getTensor().getName())) {
                errors.add("shard " + shard.getShardId() + " has duplicate state: " + state.getTensor().getName());
            } else {
                validateTensor(state.getTensor(), "state " + state.getTensor().getName(), errors);
            }
        }
        for (ExecutionEntrypoint entrypoint : safe(shard.getEntrypoints())) {
            if (entrypoint == null || blank(entrypoint.getName())) {
                errors.add("shard " + shard.getShardId() + " has unnamed entrypoint");
                continue;
            }
            if (!entrypointNames.add(entrypoint.getName())) {
                errors.add("shard " + shard.getShardId() + " has duplicate entrypoint: " + entrypoint.getName());
            }
            require(entrypoint.getCommandId(), "entrypoint " + entrypoint.getName() + " commandId", errors);
            require(entrypoint.getCapability(), "entrypoint " + entrypoint.getName() + " capability", errors);
            for (TensorContract tensor : safe(entrypoint.getInputs())) {
                validateTensor(tensor, "entrypoint " + entrypoint.getName() + " input", errors);
            }
            for (TensorContract tensor : safe(entrypoint.getOutputs())) {
                validateTensor(tensor, "entrypoint " + entrypoint.getName() + " output", errors);
            }
            for (String stateName : safe(entrypoint.getStateNames())) {
                if (!stateNames.contains(stateName)) {
                    errors.add("entrypoint " + entrypoint.getName() + " references unknown state: " + stateName);
                }
            }
        }
        if (entrypointNames.isEmpty()) {
            errors.add("shard " + shard.getShardId() + " requires an entrypoint");
        }
        for (String path : shard.getChecksums() == null ? new HashSet<String>() : shard.getChecksums().keySet()) {
            validateRelativePath(path, "checksum path", errors);
        }
    }

    private static void validateTensor(TensorContract tensor, String context, List<String> errors) {
        if (tensor == null) {
            errors.add(context + " tensor is required");
            return;
        }
        require(tensor.getName(), context + " name", errors);
        require(tensor.getDataType(), context + " dataType", errors);
        if (tensor.getShape() == null || tensor.getShape().isEmpty()) {
            errors.add(context + " shape is required");
        } else {
            for (String dimension : tensor.getShape()) {
                if (blank(dimension)) {
                    errors.add(context + " contains an empty dimension");
                }
            }
        }
    }

    private static boolean hasEntrypoint(ExecutionShardManifest shard, String name) {
        for (ExecutionEntrypoint entrypoint : safe(shard.getEntrypoints())) {
            if (entrypoint != null && entrypoint.getName() != null && entrypoint.getName().equals(name)) {
                return true;
            }
        }
        return false;
    }

    private static void detectCycles(Map<String, ExecutionNode> nodes, List<String> errors) {
        Set<String> visited = new HashSet<>();
        Set<String> visiting = new HashSet<>();
        for (String nodeId : nodes.keySet()) {
            if (hasCycle(nodeId, nodes, visited, visiting)) {
                errors.add("execution graph contains a cycle involving: " + nodeId);
                return;
            }
        }
    }

    private static boolean hasCycle(String nodeId, Map<String, ExecutionNode> nodes,
                                    Set<String> visited, Set<String> visiting) {
        if (visited.contains(nodeId)) return false;
        if (!visiting.add(nodeId)) return true;
        ExecutionNode node = nodes.get(nodeId);
        if (node != null) {
            for (String dependency : safe(node.getDependsOn())) {
                if (nodes.containsKey(dependency) && hasCycle(dependency, nodes, visited, visiting)) return true;
            }
        }
        visiting.remove(nodeId);
        visited.add(nodeId);
        return false;
    }

    private static void validateRelativePath(String value, String name, List<String> errors) {
        if (blank(value)) {
            errors.add(name + " is required");
            return;
        }
        Path path = Paths.get(value);
        if (path.isAbsolute() || value.contains("..")) {
            errors.add(name + " must be a package-relative path: " + value);
        }
    }

    private static void require(String value, String name, List<String> errors) {
        if (blank(value)) errors.add(name + " is required");
    }

    private static boolean blank(String value) {
        return value == null || value.trim().isEmpty();
    }

    private static <T> List<T> safe(List<T> values) {
        return values == null ? new ArrayList<T>() : values;
    }
}
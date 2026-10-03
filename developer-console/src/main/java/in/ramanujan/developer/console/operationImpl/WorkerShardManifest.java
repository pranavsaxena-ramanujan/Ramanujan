package in.ramanujan.developer.console.operationImpl;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

/**
 * The LLM shards this worker holds in its cache: per model, which layer indices plus the
 * embedding and output head, with the weight files behind each. It survives restarts, so a
 * device that rejoins reports the same shard offsets and the orchestrator hands that range
 * back to it instead of shipping new weights. A unit is reported only while every one of its
 * files is still cached; the orchestrator also checks the files' sizes and mtimes.
 */
final class WorkerShardManifest {
    static final int MAX_MODELS = 16;
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final ObjectMapper SORTED = new ObjectMapper().enable(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS);

    private final Path file;
    private final String gateway;

    WorkerShardManifest(Path root, String serverUrl) {
        this.file = root.toAbsolutePath().resolve("shards.json");
        // Only the room that shipped the weights may claim them; the bearer URL itself is never stored.
        this.gateway = sha256(serverUrl);
    }

    /** Model identity shared with the orchestrator: SHA-256 of the key-sorted graph "hyper" JSON. */
    static String modelKey(Object hyper) {
        try {
            return sha256(SORTED.writeValueAsString(hyper));
        } catch (IOException e) {
            throw new IllegalArgumentException("graph hyper is not serializable", e);
        }
    }

    /** Records the units of a graph whose weights were just cached (server paths, before localization). */
    @SuppressWarnings("unchecked")
    synchronized void record(Map<String, Object> graph) throws IOException {
        if (graph == null || graph.get("hyper") == null) return;
        Map<String, List<String>> units = new LinkedHashMap<>();
        String rope = fileOf(graph.get("rope_freqs"));
        if (graph.get("embed") != null) units.put("embed", files(graph.get("embed")));
        if (graph.get("layers") instanceof List) {
            for (Object entry : (List<Object>) graph.get("layers")) {
                if (!(entry instanceof Map) || !(((Map<String, Object>) entry).get("index") instanceof Number)) return;
                List<String> layerFiles = files(((Map<String, Object>) entry).get("tensors"));
                if (rope != null) layerFiles.add(rope);
                units.put("L" + ((Number) ((Map<String, Object>) entry).get("index")).intValue(), layerFiles);
            }
        }
        if (graph.get("head") != null) units.put("head", files(graph.get("head")));
        if (units.isEmpty()) return;
        Map<String, Object> manifest = load();
        Map<String, Object> models = (Map<String, Object>) manifest.get("models");
        String id = gateway + ":" + modelKey(graph.get("hyper"));
        Map<String, Object> model = (Map<String, Object>) models.get(id);
        if (model == null) {
            model = new LinkedHashMap<>();
            model.put("gateway", gateway);
            model.put("model", modelKey(graph.get("hyper")));
            model.put("units", new LinkedHashMap<String, Object>());
            models.put(id, model);
        }
        ((Map<String, Object>) model.get("units")).putAll(units);
        model.put("lastUsed", System.currentTimeMillis());
        while (models.size() > MAX_MODELS) {
            String oldest = null;
            for (Map.Entry<String, Object> entry : models.entrySet()) {
                long used = ((Number) ((Map<String, Object>) entry.getValue()).get("lastUsed")).longValue();
                if (oldest == null || used < ((Number) ((Map<String, Object>) models.get(oldest)).get("lastUsed")).longValue()) {
                    oldest = entry.getKey();
                }
            }
            models.remove(oldest);
        }
        Path temp = file.resolveSibling("shards.json.tmp");
        Files.createDirectories(file.getParent());
        MAPPER.writeValue(temp.toFile(), manifest);
        Files.move(temp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    /**
     * Shards still fully cached, for the capacity ping:
     * {model, embed, head, layers: [[start, end), ...], bytes}. {@code cached} maps server path to bytes.
     */
    @SuppressWarnings("unchecked")
    synchronized List<Map<String, Object>> report(Map<String, Long> cached) {
        List<Map<String, Object>> shards = new ArrayList<>();
        Map<String, Object> models;
        try {
            models = (Map<String, Object>) load().get("models");
        } catch (IOException | RuntimeException e) {
            return shards;
        }
        for (Object value : models.values()) {
            Map<String, Object> model = (Map<String, Object>) value;
            if (!gateway.equals(model.get("gateway"))) continue;
            boolean embed = false, head = false;
            TreeSet<Integer> layers = new TreeSet<>();
            Map<String, Long> counted = new HashMap<>();
            for (Map.Entry<String, Object> unit : ((Map<String, Object>) model.get("units")).entrySet()) {
                List<String> paths = (List<String>) unit.getValue();
                if (paths.isEmpty() || !cached.keySet().containsAll(paths)) continue;
                for (String path : paths) counted.put(path, cached.get(path));
                if ("embed".equals(unit.getKey())) embed = true;
                else if ("head".equals(unit.getKey())) head = true;
                else layers.add(Integer.parseInt(unit.getKey().substring(1)));
            }
            if (!embed && !head && layers.isEmpty()) continue;
            List<List<Integer>> ranges = new ArrayList<>();
            for (int index : layers) {
                List<Integer> last = ranges.isEmpty() ? null : ranges.get(ranges.size() - 1);
                if (last != null && last.get(1) == index) last.set(1, index + 1);
                else ranges.add(new ArrayList<>(java.util.Arrays.asList(index, index + 1)));
            }
            long bytes = 0;
            for (long size : counted.values()) bytes += size;
            Map<String, Object> shard = new LinkedHashMap<>();
            shard.put("model", model.get("model"));
            shard.put("embed", embed);
            shard.put("head", head);
            shard.put("layers", ranges);
            shard.put("bytes", bytes);
            shards.add(shard);
        }
        return shards;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> load() throws IOException {
        Map<String, Object> manifest = Files.isRegularFile(file) ? MAPPER.readValue(file.toFile(), Map.class) : null;
        if (manifest == null || !(manifest.get("models") instanceof Map)) {
            manifest = new LinkedHashMap<>();
            manifest.put("version", 1);
            manifest.put("models", new LinkedHashMap<String, Object>());
        }
        return manifest;
    }

    private static String fileOf(Object node) {
        return node instanceof Map && ((Map<?, ?>) node).get("file") instanceof String ? (String) ((Map<?, ?>) node).get("file") : null;
    }

    private static List<String> files(Object node) {
        List<String> result = new ArrayList<>();
        collect(node, result);
        Collections.sort(result);
        return result;
    }

    private static void collect(Object node, List<String> files) {
        if (node instanceof Map) {
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) node).entrySet()) {
                if ("file".equals(entry.getKey()) && entry.getValue() instanceof String) files.add((String) entry.getValue());
                else collect(entry.getValue(), files);
            }
        } else if (node instanceof List) {
            for (Object item : (List<?>) node) collect(item, files);
        }
    }

    private static String sha256(String value) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder();
            for (byte b : digest) hex.append(String.format("%02x", b));
            return hex.toString();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }
}

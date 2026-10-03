package in.ramanujan.orchestrator.data.dao;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import in.ramanujan.db.layer.utils.CapacityStore;
import in.ramanujan.orchestrator.base.NativeSessionKey;
import io.vertx.core.Future;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.nio.file.*;
import java.util.*;

@Component
public class CapacityDao {
    public static final long FRESH_MILLIS = 45_000;
    private static final long HEADROOM = 256L * 1024 * 1024;
    // Driver context, kernels and fragmentation; the rest of VRAM can hold one large resident shard.
    private static final long GPU_RESERVE = 1024L * 1024 * 1024;
    @Autowired private CapacityStore store;
    private final ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS);

    public boolean enabled() {
        return "1".equals(System.getenv("RAMANUJAN_CAPACITY_AWARE"));
    }

    public static String cluster(Map<String, Object> request) {
        return string(request, "clusterId", 100);
    }

    private static String string(Map<String, Object> request, String key, int max) {
        Object value = request.get(key);
        if (!(value instanceof String) || ((String) value).isEmpty() || ((String) value).length() > max) {
            throw new IllegalArgumentException(key + " must be a nonempty string of at most " + max + " characters");
        }
        return (String) value;
    }

    private static long number(Map<String, Object> request, String key) {
        Object value = request.get(key);
        if (!(value instanceof Number) || ((Number) value).doubleValue() < 0
                || ((Number) value).doubleValue() > 9_007_199_254_740_991d
                || !Double.isFinite(((Number) value).doubleValue())
                || ((Number) value).doubleValue() != ((Number) value).longValue()) {
            throw new IllegalArgumentException(key + " must be a nonnegative integer");
        }
        return ((Number) value).longValue();
    }

    private static long optional(Map<String, Object> request, String key) {
        return request.get(key) == null ? 0 : number(request, key);
    }

    private static boolean bool(Map<String, Object> request, String key) {
        if (!(request.get(key) instanceof Boolean)) throw new IllegalArgumentException(key + " must be a boolean");
        return Boolean.TRUE.equals(request.get(key));
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> objects(Map<String, Object> request, String key, int maximum) {
        Object value = request.get(key);
        if (!(value instanceof List) || ((List<?>) value).size() > maximum) {
            throw new IllegalArgumentException(key + " must be an array of at most " + maximum + " objects");
        }
        for (Object entry : (List<?>) value) if (!(entry instanceof Map)) {
            throw new IllegalArgumentException(key + " must contain objects");
        }
        return (List<Map<String, Object>>) value;
    }

    public Future<Map<String, Object>> report(Map<String, Object> request) {
        try {
            if (number(request, "schemaVersion") != 1) throw new IllegalArgumentException("Unsupported capacity schema");
            String cluster = cluster(request);
            String host = string(request, "hostId", 200);
            String boot = string(request, "bootId", 128);
            long sequence = number(request, "sequence");
            if (sequence < 1) throw new IllegalArgumentException("sequence must be positive");
            Map<String, Object> snapshot = new LinkedHashMap<>();
            snapshot.put("hostId", host);
            snapshot.put("bootId", boot);
            snapshot.put("sequence", sequence);
            for (String metric : Arrays.asList("ramTotalBytes", "ramAvailableBytes", "ramAllocatedBytes",
                    "gpuTotalBytes", "gpuAvailableBytes", "gpuAllocatedBytes", "gpuResidentBytes",
                    "gpuMaxAllocationBytes", "diskAvailableBytes", "cacheBytes", "activeSessions",
                    "sessionLimit", "capacitySessionLimit", "affinityLimit")) {
                snapshot.put(metric, request.get(metric) == null ? null : number(request, metric));
            }
            snapshot.put("unifiedMemory", bool(request, "unifiedMemory"));
            snapshot.put("acceptingNewWork", bool(request, "acceptingNewWork"));
            if (request.containsKey("nativeReady")) snapshot.put("nativeReady", bool(request, "nativeReady"));
            if (request.containsKey("cachedFiles")) {
                List<Map<String, Object>> cached = new ArrayList<>();
                for (Map<String, Object> file : objects(request, "cachedFiles", 4096)) {
                    cached.add(map("path", string(file, "path", 4096), "bytes", number(file, "bytes"),
                            "mtime", number(file, "mtime")));
                }
                snapshot.put("cachedFiles", cached);
            }
            if (request.containsKey("cachedShards")) snapshot.put("cachedShards", cachedShards(request));
            for (String prefix : Arrays.asList("ram", "gpu")) {
                if (snapshot.get(prefix + "AvailableBytes") != null && snapshot.get(prefix + "TotalBytes") != null
                        && number(snapshot, prefix + "AvailableBytes") > number(snapshot, prefix + "TotalBytes")) {
                    throw new IllegalArgumentException(prefix + " available memory exceeds total memory");
                }
            }
            return store.atomic(cluster, state -> {
                Map<String, Object> previous = state.hosts.get(host);
                Set<String> retired = new LinkedHashSet<>();
                if (previous != null) {
                    Object oldRetired = previous.get("retiredBootIds");
                    if (oldRetired instanceof List) for (Object id : (List<?>) oldRetired) retired.add((String) id);
                    if (retired.contains(boot)) return map("accepted", false, "reason", "retired worker boot");
                    if (boot.equals(previous.get("bootId")) && sequence <= number(previous, "sequence")) {
                        return map("accepted", false, "reason", "out-of-order sample");
                    }
                    if (!boot.equals(previous.get("bootId"))) retired.add((String) previous.get("bootId"));
                    if (retired.size() > 256) throw new IllegalStateException("Worker boot history requires administrator cleanup");
                }
                snapshot.put("retiredBootIds", new ArrayList<>(retired));
                snapshot.put("receivedAt", System.currentTimeMillis());
                state.hosts.put(host, snapshot);
                return map("accepted", true);
            });
        } catch (Exception error) { return Future.failedFuture(error); }
    }

    /** Shards a worker reports as cached: {model, embed, head, layers: [[start, end), ...], bytes}. */
    private static List<Map<String, Object>> cachedShards(Map<String, Object> request) {
        List<Map<String, Object>> shards = new ArrayList<>();
        for (Map<String, Object> shard : objects(request, "cachedShards", 64)) {
            Object raw = shard.get("layers");
            if (!(raw instanceof List) || ((List<?>) raw).size() > 1024) {
                throw new IllegalArgumentException("cachedShards layers must be an array of at most 1024 ranges");
            }
            List<Object> ranges = new ArrayList<>();
            for (Object range : (List<?>) raw) {
                if (!(range instanceof List) || ((List<?>) range).size() != 2) {
                    throw new IllegalArgumentException("cachedShards layer ranges must be [start, end)");
                }
                Map<String, Object> bounds = map("start", ((List<?>) range).get(0), "end", ((List<?>) range).get(1));
                long start = number(bounds, "start"), end = number(bounds, "end");
                if (end <= start || end > 1 << 20) throw new IllegalArgumentException("cachedShards layer range is empty or too large");
                ranges.add(Arrays.asList(start, end));
            }
            shards.add(map("model", string(shard, "model", 64), "embed", bool(shard, "embed"),
                    "head", bool(shard, "head"), "layers", ranges, "bytes", number(shard, "bytes")));
        }
        return shards;
    }

    /** Model identity shared with workers: SHA-256 of the key-sorted graph "hyper" JSON. */
    String modelKey(Object hyper) throws java.io.IOException {
        try {
            byte[] digest = java.security.MessageDigest.getInstance("SHA-256")
                    .digest(mapper.writeValueAsString(hyper).getBytes(java.nio.charset.StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder();
            for (byte b : digest) hex.append(String.format("%02x", b));
            return hex.toString();
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    /** Bytes of {@code stage}'s files the host does not already hold with the same size and mtime. */
    private static long downloadBytes(Map<String, Object> host, Map<String, Object> stage) {
        Map<String, Map<String, Object>> cached = cachedFiles(host);
        long bytes = 0;
        for (Map<String, Object> file : objects(stage, "files", 1 << 16)) {
            if (!isCached(cached, file)) bytes = Math.addExact(bytes, number(file, "bytes"));
        }
        return bytes;
    }

    private static Map<String, Map<String, Object>> cachedFiles(Map<String, Object> host) {
        Map<String, Map<String, Object>> cached = new HashMap<>();
        if (host != null && host.containsKey("cachedFiles")) {
            for (Map<String, Object> file : objects(host, "cachedFiles", 4096)) cached.put((String) file.get("path"), file);
        }
        return cached;
    }

    private static boolean isCached(Map<String, Map<String, Object>> cached, Map<String, Object> file) {
        Map<String, Object> present = cached.get(file.get("path"));
        return present != null && number(file, "bytes") == number(present, "bytes")
                && number(file, "mtime") == number(present, "mtime");
    }

    public Future<List<Map<String, Object>>> snapshots(String cluster) {
        return store.atomic(cluster, state -> {
            List<Map<String, Object>> result = new ArrayList<>();
            for (Map<String, Object> host : state.hosts.values()) {
                Map<String, Object> copy = new LinkedHashMap<>(host);
                copy.remove("retiredBootIds");
                copy.remove("cachedFiles");
                copy.put("fresh", System.currentTimeMillis() - number(host, "receivedAt") < FRESH_MILLIS);
                boolean eligible = eligible(host, System.currentTimeMillis());
                copy.put("eligible", eligible);
                if (eligible) {
                    long[] committed = committed(host, state);
                    copy.put("gpuBudgetBytes", gpuBudget(host));
                    copy.put("ramBudgetBytes", ramBudget(host));
                    copy.put("gpuCommittedBytes", committed[0]);
                    copy.put("ramCommittedBytes", committed[1]);
                }
                result.add(copy);
            }
            return result;
        });
    }

    public Future<Set<String>> reservedHosts(String cluster) {
        return store.atomic(cluster, state -> {
            Set<String> hosts = new HashSet<>();
            for (Map<String, Object> plan : state.plans.values()) if ("ACTIVE".equals(plan.get("state"))) {
                for (Map<String, Object> stage : stages(plan)) hosts.add((String) stage.get("hostId"));
            }
            return hosts;
        });
    }

    @SuppressWarnings("unchecked")
    public Future<Map<String, Object>> reserve(Map<String, Object> request) {
        try {
            String cluster = cluster(request);
            String mode = string(request, "weights", 16);
            if (!Arrays.asList("auto", "resident", "stream").contains(mode)) throw new IllegalArgumentException("Invalid weights mode");
            String placement = request.get("placement") == null ? "search" : string(request, "placement", 16);
            if (!Arrays.asList("search", "vram").contains(placement)) throw new IllegalArgumentException("Invalid placement");
            boolean vram = "vram".equals(placement);
            if (vram && "stream".equals(mode)) throw new IllegalArgumentException("VRAM placement sizes resident shards; use auto or resident");
            boolean dryRun = request.get("dryRun") != null && bool(request, "dryRun");
            if (dryRun && !vram) throw new IllegalArgumentException("dryRun requires vram placement");
            List<Map<String, Object>> stages = new ArrayList<>();
            Set<String> bindings = new HashSet<>();
            for (Map<String, Object> input : objects(request, "stages", 1024)) {
                Map<String, Object> stage = new LinkedHashMap<>();
                String affinity = string(input, "affinity", 200);
                String session = string(input, "session", 200);
                String binding = NativeSessionKey.binding(cluster, affinity, session);
                if (!bindings.add(binding)) throw new IllegalArgumentException("Duplicate planned binding");
                stage.put("affinity", affinity);
                stage.put("session", session);
                stage.put("binding", binding);
                for (String key : Arrays.asList("weightBytes", "streamWorkingBytes", "stateBytes", "scratchBytes")) {
                    stage.put(key, number(input, key));
                }
                stage.put("maxAllocationBytes", optional(input, "maxAllocationBytes"));
                if (!(input.get("graph") instanceof Map)) throw new IllegalArgumentException("Planned stage graph required");
                stage.put("graph", mapper.readValue(mapper.writeValueAsBytes(input.get("graph")), Map.class));
                if (number(stage, "weightBytes") > 0 && number(stage, "streamWorkingBytes") == 0) {
                    throw new IllegalArgumentException("Streaming working set must be specified");
                }
                List<Map<String, Object>> files = new ArrayList<>();
                Set<String> paths = new HashSet<>();
                long totalFileBytes = 0;
                for (Map<String, Object> file : objects(input, "files", 4096)) {
                    String path = string(file, "path", 4096);
                    if (!paths.add(path)) continue;
                    Path local = Paths.get(path);
                    if (!Files.isRegularFile(local)) throw new IllegalArgumentException("Planned weight file is missing");
                    long bytes = Files.size(local);
                    if (bytes != number(file, "bytes")) throw new IllegalArgumentException("Weight file size changed");
                    long mtime = Files.getLastModifiedTime(local).toMillis();
                    if (file.get("mtime") != null && mtime != number(file, "mtime")) {
                        throw new IllegalArgumentException("Weight file changed");
                    }
                    totalFileBytes = Math.addExact(totalFileBytes, bytes);
                    files.add(map("path", path, "bytes", bytes, "mtime", mtime));
                }
                if (totalFileBytes > number(stage, "weightBytes")) {
                    throw new IllegalArgumentException("weightBytes understates declared files");
                }
                if (!paths.containsAll(graphFiles(stage.get("graph")))) {
                    throw new IllegalArgumentException("Every graph weight file must be declared for disk admission");
                }
                stage.put("files", files);
                if (vram) {
                    // Per-session parts let the orchestrator bound a merged session without re-estimating it.
                    stage.put("sharedScratchBytes", number(input, "sharedScratchBytes"));
                    stage.put("streamItemBytes", number(input, "streamItemBytes"));
                    stage.put("streamSharedBytes", number(input, "streamSharedBytes"));
                    if (number(stage, "sharedScratchBytes") > number(stage, "scratchBytes")
                            || Math.addExact(number(stage, "streamItemBytes"), number(stage, "streamSharedBytes"))
                            != number(stage, "streamWorkingBytes")) {
                        throw new IllegalArgumentException("Per-session capacity parts are inconsistent");
                    }
                }
                stages.add(stage);
            }
            if (stages.isEmpty()) throw new IllegalArgumentException("stages are required");
            return store.atomic(cluster, state -> {
                for (Map<String, Object> plan : state.plans.values()) {
                    if ("RELEASED".equals(plan.get("state"))) continue;
                    for (Map<String, Object> stage : stages(plan)) if (bindings.contains(stage.get("binding"))) {
                        throw new IllegalArgumentException("Binding already belongs to a capacity plan");
                    }
                }
                List<Map<String, Object>> hosts = new ArrayList<>();
                long now = System.currentTimeMillis();
                for (Map<String, Object> host : state.hosts.values()) if (eligible(host, now)) hosts.add(host);
                // Largest usable VRAM first, so the biggest device receives the biggest shard.
                hosts.sort(Comparator.<Map<String, Object>>comparingLong(CapacityDao::residentBudget).reversed()
                        .thenComparing(Comparator.<Map<String, Object>>comparingLong(
                                host -> number(host, "diskAvailableBytes")).reversed())
                        .thenComparing(host -> (String) host.get("hostId")));
                if (hosts.size() > 128) hosts = new ArrayList<>(hosts.subList(0, 128));
                List<Map<String, Object>> assigned = null;
                if (vram) assigned = new VramPlacement(stages, hosts, state, mode).place();
                if (!vram && !"stream".equals(mode)) assigned = allocate(stages, hosts, state, "resident", 0, 0, new HashSet<>());
                if (!vram && assigned == null && !"resident".equals(mode)) {
                    assigned = allocate(stages, hosts, state, "stream", 0, 0, new HashSet<>());
                }
                if (assigned == null) throw new IllegalStateException(
                        "Connected devices lack fresh capacity, disk space or streaming working memory for this run");
                List<Map<String, Object>> reply = new ArrayList<>();
                for (Map<String, Object> stage : assigned) {
                    Map<String, Object> entry = map("affinity", stage.get("affinity"), "session", stage.get("session"),
                            "hostId", stage.get("hostId"), "weights", stage.get("weights"));
                    if (vram) {
                        entry.put("pieces", stage.get("pieces"));
                        entry.put("graph", stage.get("graph"));
                        entry.put("weightBytes", stage.get("weightBytes"));
                        Map<String, Object> host = state.hosts.get(stage.get("hostId"));
                        entry.put("deviceBytes", required(host, Collections.singletonList(stage))[0]);
                        entry.put("gpuBudgetBytes", gpuBudget(host));
                        entry.put("ramBudgetBytes", ramBudget(host));
                        entry.put("gpuTotalBytes", host.get("gpuTotalBytes"));
                        entry.put("unifiedMemory", host.get("unifiedMemory"));
                        entry.put("downloadBytes", downloadBytes(host, stage));
                    }
                    reply.add(entry);
                }
                if (dryRun) return map("dryRun", true, "stages", reply);
                String id = UUID.randomUUID().toString();
                Map<String, Object> plan = map("planId", id, "state", "ACTIVE", "createdAt", now, "stages", assigned);
                state.plans.put(id, plan);
                return map("planId", id, "stages", reply);
            });
        } catch (Exception error) { return Future.failedFuture(error); }
    }

    private static boolean eligible(Map<String, Object> host, long now) {
        return now - number(host, "receivedAt") < FRESH_MILLIS && Boolean.TRUE.equals(host.get("acceptingNewWork"))
                && !Boolean.FALSE.equals(host.get("nativeReady"))
                && host.get("ramAvailableBytes") != null && host.get("ramTotalBytes") != null
                && host.get("gpuTotalBytes") != null && host.get("diskAvailableBytes") != null;
    }

    static long ramBudget(Map<String, Object> host) {
        long ramAvailable = Math.addExact(number(host, "ramAvailableBytes"), optional(host, "ramAllocatedBytes"));
        if (Boolean.TRUE.equals(host.get("unifiedMemory"))) {
            ramAvailable = Math.addExact(ramAvailable, optional(host, "gpuAllocatedBytes"));
        }
        return Math.max(0, Math.min(number(host, "ramTotalBytes") / 4 * 3, ramAvailable) - HEADROOM);
    }

    static long gpuBudget(Map<String, Object> host) {
        long total = number(host, "gpuTotalBytes");
        long budget = total - Math.max(GPU_RESERVE, total / 10);
        if (host.get("gpuAvailableBytes") != null) {
            budget = Math.min(budget, Math.addExact(number(host, "gpuAvailableBytes"), host.get("gpuAllocatedBytes") != null
                    ? number(host, "gpuAllocatedBytes") : optional(host, "gpuResidentBytes")));
        }
        return Math.max(0, budget - HEADROOM);
    }

    /** Bytes one resident shard may occupy: unified-memory devices share the RAM budget. */
    static long residentBudget(Map<String, Object> host) {
        long gpu = gpuBudget(host);
        return Boolean.TRUE.equals(host.get("unifiedMemory")) ? Math.min(gpu, ramBudget(host)) : gpu;
    }

    /** {device bytes, host RAM bytes} already reserved on this host by active plans. */
    private static long[] committed(Map<String, Object> host, CapacityStore.State state) {
        List<Map<String, Object>> active = new ArrayList<>();
        for (Map<String, Object> plan : state.plans.values()) if ("ACTIVE".equals(plan.get("state"))) {
            for (Map<String, Object> stage : stages(plan)) if (host.get("hostId").equals(stage.get("hostId"))) active.add(stage);
        }
        return required(host, active);
    }

    private static long[] required(Map<String, Object> host, List<Map<String, Object>> stages) {
        long state = 0, weights = 0, workspace = 0;
        for (Map<String, Object> stage : stages) {
            state = Math.addExact(state, number(stage, "stateBytes"));
            if ("resident".equals(stage.get("weights"))) weights = Math.addExact(weights, number(stage, "weightBytes"));
            long stream = "stream".equals(stage.get("weights")) ? number(stage, "streamWorkingBytes") : 0;
            // Each session keeps its own prefetch window and workspace even while another stage executes.
            workspace = Math.addExact(workspace, Math.addExact(stream, number(stage, "scratchBytes")));
        }
        long device = Math.addExact(Math.addExact(state, weights), workspace);
        // A discrete GPU reads resident weights through a reused staging buffer and keeps no host copy.
        long ram = Boolean.TRUE.equals(host.get("unifiedMemory")) ? device : Math.addExact(state, workspace);
        return new long[]{device, ram};
    }

    /**
     * Orchestrator-side whole-layer sharding from the latest capacity pings. Devices are visited by
     * free resident budget (VRAM, and RAM on unified memory), largest first; each takes the longest
     * run of consecutive pieces that stays resident, so the biggest device gets the biggest shard
     * and the model is split only when it must be. If memory cannot hold everything, one device
     * also streams a run of middle layers ([resident][stream][resident head]); the device that
     * streams the fewest bytes is chosen. Consecutive pieces merge into one session per run, with
     * conservative merged budgets: weights/state add up and per-session overheads count once.
     * Devices that report a cached shard of this model also get a candidate plan that keeps them on
     * those offsets and shards only the remainder; the plan streaming the fewest bytes wins, then
     * the one needing the fewest new downloads.
     */
    private final class VramPlacement {
        private final List<Map<String, Object>> pieces;
        private final List<Map<String, Object>> order;
        private final CapacityStore.State state;
        private final String mode;
        private final Map<String, Map<String, Object>> merged = new HashMap<>();

        VramPlacement(List<Map<String, Object>> pieces, List<Map<String, Object>> hosts, CapacityStore.State state, String mode) {
            this.pieces = pieces;
            this.state = state;
            this.mode = mode;
            this.order = new ArrayList<>(hosts);
            order.sort(Comparator.<Map<String, Object>>comparingLong(this::freeResident).reversed()
                    .thenComparing(host -> (String) host.get("hostId")));
        }

        private long freeResident(Map<String, Object> host) {
            long[] used = committed(host, state);
            long gpu = gpuBudget(host) - used[0];
            return Boolean.TRUE.equals(host.get("unifiedMemory")) ? Math.min(gpu, ramBudget(host) - used[1]) : gpu;
        }

        private Map<String, Object> group(int start, int end, Map<String, Object> host, String weights) {
            String key = start + ":" + end + ":" + host.get("hostId") + ":" + weights;
            Map<String, Object> cached = merged.get(key);
            if (cached != null) return cached;
            Map<String, Object> first = pieces.get(start);
            long weight = 0, stateBytes = 0, scratch = 0, shared = 0, item = 0, streamShared = 0, embedStream = 0, maxAllocation = 0;
            Map<String, Map<String, Object>> files = new LinkedHashMap<>();
            Map<String, Object> graph = new LinkedHashMap<>(graph(first));
            List<Object> layers = new ArrayList<>();
            graph.remove("embed");
            graph.remove("head");
            graph.remove("rope_freqs");
            for (int index = start; index < end; index++) {
                Map<String, Object> piece = pieces.get(index);
                Map<String, Object> pieceGraph = graph(piece);
                weight = Math.addExact(weight, number(piece, "weightBytes"));
                stateBytes = Math.addExact(stateBytes, number(piece, "stateBytes"));
                scratch = Math.addExact(scratch, number(piece, "scratchBytes") - number(piece, "sharedScratchBytes"));
                shared = Math.max(shared, number(piece, "sharedScratchBytes"));
                item = Math.max(item, number(piece, "streamItemBytes"));
                // The embedding row buffer is separate from layer windows; rope/staging are shared.
                if (pieceGraph.containsKey("embed")) embedStream = Math.addExact(embedStream, number(piece, "streamSharedBytes"));
                else streamShared = Math.max(streamShared, number(piece, "streamSharedBytes"));
                maxAllocation = Math.max(maxAllocation, optional(piece, "maxAllocationBytes"));
                for (Map<String, Object> file : objects(piece, "files", 4096)) files.put((String) file.get("path"), file);
                for (String part : Arrays.asList("embed", "head", "rope_freqs")) {
                    if (pieceGraph.containsKey(part)) graph.put(part, pieceGraph.get(part));
                }
                if (pieceGraph.get("layers") instanceof List) layers.addAll((List<?>) pieceGraph.get("layers"));
            }
            graph.put("layers", layers);
            graph.put("weights", weights);
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("affinity", first.get("affinity"));
            result.put("session", first.get("session"));
            result.put("binding", first.get("binding"));
            result.put("weightBytes", weight);
            result.put("stateBytes", stateBytes);
            result.put("scratchBytes", Math.addExact(scratch, shared));
            result.put("streamWorkingBytes", Math.addExact(Math.addExact(item, streamShared), embedStream));
            result.put("maxAllocationBytes", maxAllocation);
            result.put("files", new ArrayList<>(files.values()));
            result.put("graph", graph);
            result.put("pieces", Arrays.asList(start, end));
            result.put("hostId", host.get("hostId"));
            result.put("bootId", host.get("bootId"));
            result.put("weights", weights);
            merged.put(key, result);
            return result;
        }

        private boolean fitsAll(Map<String, Object> host, List<Map<String, Object>> sessions) {
            return fits(host, sessions, state);
        }

        private int longest(Map<String, Object> host, int start, List<Map<String, Object>> extra, int limit) {
            int low = start, high = limit;
            while (low < high) {
                int middle = (low + high + 1) / 2;
                List<Map<String, Object>> sessions = new ArrayList<>(extra);
                sessions.add(group(start, middle, host, "resident"));
                if (fitsAll(host, sessions)) low = middle;
                else high = middle - 1;
            }
            return low;
        }

        private List<Map<String, Object>> streaming(Map<String, Object> host, int start) {
            List<Map<String, Object>> chosen = null;
            long chosenBytes = Long.MAX_VALUE;
            int count = pieces.size();
            for (int tail = 0; tail <= 1; tail++) {
                int stop = count - tail;
                if (stop - start < 1) continue;
                List<Map<String, Object>> kept = tail == 1
                        ? Collections.singletonList(group(stop, count, host, "resident")) : Collections.emptyList();
                int end = longest(host, start, kept, stop - 1);
                List<Map<String, Object>> sessions = null;
                for (; end >= start; end--) {
                    sessions = new ArrayList<>();
                    if (end > start) sessions.add(group(start, end, host, "resident"));
                    sessions.add(group(end, stop, host, "stream"));
                    sessions.addAll(kept);
                    if (fitsAll(host, sessions)) break;
                }
                if (end < start) continue;
                long streamed = number(group(end, stop, host, "stream"), "weightBytes");
                // On ties keep the output head resident: streaming it would widen every prefetch window.
                if (streamed <= chosenBytes) { chosenBytes = streamed; chosen = sessions; }
            }
            return chosen;
        }

        List<Map<String, Object>> place() throws java.io.IOException {
            List<List<Map<String, Object>>> candidates = new ArrayList<>();
            List<Map<String, Object>> streamers = new ArrayList<>();
            streamers.add(null);
            if ("auto".equals(mode)) streamers.addAll(order);
            for (Map<String, Object> streamer : streamers) {
                List<Map<String, Object>> sequence = new ArrayList<>();
                for (Map<String, Object> host : order) if (host != streamer) sequence.add(host);
                if (streamer != null) sequence.add(streamer);
                int cursor = 0;
                List<Map<String, Object>> groups = new ArrayList<>();
                for (Map<String, Object> host : sequence) {
                    if (cursor == pieces.size()) break;
                    if (host == streamer) {
                        List<Map<String, Object>> planned = streaming(host, cursor);
                        if (planned == null) break;
                        groups.addAll(planned);
                        cursor = pieces.size();
                    } else {
                        int end = longest(host, cursor, Collections.emptyList(), pieces.size());
                        if (end > cursor) {
                            groups.add(group(cursor, end, host, "resident"));
                            cursor = end;
                        }
                    }
                }
                if (cursor != pieces.size()) continue;
                candidates.add(groups);
                if (score(groups)[0] == 0) break;
            }
            Map<Map<String, Object>, int[]> anchors = anchors();
            if (!anchors.isEmpty()) {
                List<Map<String, Object>> anchored = new ArrayList<>(anchors.keySet());
                anchored.sort(Comparator.comparingInt(host -> anchors.get(host)[0]));
                List<Map<String, Object>> cacheStreamers = new ArrayList<>();
                cacheStreamers.add(null);
                if ("auto".equals(mode)) {
                    for (Map<String, Object> host : order) if (!anchors.containsKey(host)) cacheStreamers.add(host);
                    cacheStreamers.add(anchored.get(anchored.size() - 1));
                }
                for (Map<String, Object> streamer : cacheStreamers) {
                    List<Map<String, Object>> groups = anchoredWalk(anchored, anchors, streamer);
                    if (groups != null) candidates.add(groups);
                }
            }
            // Streamed bytes cost disk I/O on every token; downloads are paid once.
            List<Map<String, Object>> best = null;
            long[] bestScore = null;
            for (List<Map<String, Object>> groups : candidates) {
                long[] score = score(groups);
                if (bestScore == null || score[0] < bestScore[0] || (score[0] == bestScore[0] && score[1] < bestScore[1])) {
                    best = groups;
                    bestScore = score;
                }
            }
            if (best == null) return null;
            List<Map<String, Object>> result = new ArrayList<>();
            for (Map<String, Object> group : best) result.add(new LinkedHashMap<>(group));
            return result;
        }

        /** {streamed weight bytes, bytes devices must newly download}. */
        private long[] score(List<Map<String, Object>> groups) {
            long streamed = 0, download = 0;
            for (Map<String, Object> group : groups) {
                if ("stream".equals(group.get("weights"))) streamed = Math.addExact(streamed, number(group, "weightBytes"));
                download = Math.addExact(download, downloadBytes(state.hosts.get(group.get("hostId")), group));
            }
            return new long[]{streamed, download};
        }

        /**
         * Each device's cached run of this model: the longest run of consecutive pieces inside a
         * shard it reports whose files it still holds unchanged, trimmed to what fits resident now.
         * Bigger devices claim first if two devices report the same layers.
         */
        private Map<Map<String, Object>, int[]> anchors() throws java.io.IOException {
            Map<Map<String, Object>, int[]> result = new IdentityHashMap<>();
            Object hyper = graph(pieces.get(0)).get("hyper");
            if (hyper == null) return result;
            String model = modelKey(hyper);
            boolean[] taken = new boolean[pieces.size()];
            for (Map<String, Object> host : order) {
                if (!(host.get("cachedShards") instanceof List)) continue;
                Map<String, Map<String, Object>> files = cachedFiles(host);
                int bestStart = -1, bestEnd = -1;
                long bestBytes = 0;
                for (Map<String, Object> shard : objects(host, "cachedShards", 64)) {
                    if (!model.equals(shard.get("model"))) continue;
                    int runStart = -1;
                    long runBytes = 0;
                    for (int index = 0; index <= pieces.size(); index++) {
                        if (index < pieces.size() && !taken[index] && holds(shard, files, pieces.get(index))) {
                            if (runStart < 0) { runStart = index; runBytes = 0; }
                            runBytes = Math.addExact(runBytes, number(pieces.get(index), "weightBytes"));
                        } else if (runStart >= 0) {
                            if (runBytes > bestBytes) { bestStart = runStart; bestEnd = index; bestBytes = runBytes; }
                            runStart = -1;
                        }
                    }
                }
                if (bestStart < 0) continue;
                int end = longest(host, bestStart, Collections.emptyList(), bestEnd);
                if (end == bestStart) continue;
                for (int index = bestStart; index < end; index++) taken[index] = true;
                result.put(host, new int[]{bestStart, end});
            }
            return result;
        }

        private boolean holds(Map<String, Object> shard, Map<String, Map<String, Object>> files, Map<String, Object> piece) {
            Map<String, Object> pieceGraph = graph(piece);
            if (pieceGraph.containsKey("embed") && !Boolean.TRUE.equals(shard.get("embed"))) return false;
            if (pieceGraph.containsKey("head") && !Boolean.TRUE.equals(shard.get("head"))) return false;
            if (pieceGraph.get("layers") instanceof List) {
                for (Object layer : (List<?>) pieceGraph.get("layers")) {
                    Object index = layer instanceof Map ? ((Map<?, ?>) layer).get("index") : null;
                    if (!(index instanceof Number)) return false;
                    boolean inside = false;
                    for (Object range : (List<?>) shard.get("layers")) {
                        List<?> bounds = (List<?>) range;
                        long value = ((Number) index).longValue();
                        if (value >= ((Number) bounds.get(0)).longValue() && value < ((Number) bounds.get(1)).longValue()) inside = true;
                    }
                    if (!inside) return false;
                }
            }
            for (Map<String, Object> file : objects(piece, "files", 4096)) if (!isCached(files, file)) return false;
            return true;
        }

        /**
         * Keeps every anchored device on its cached offsets and shards the remaining gaps big to
         * small across the other devices. Each device still holds one contiguous range: an anchored
         * device starts earlier if the gap before it cannot be filled, and extends up to the next
         * anchor. The optional streamer takes whatever is left at the end of the model.
         */
        private List<Map<String, Object>> anchoredWalk(List<Map<String, Object>> anchored,
                Map<Map<String, Object>, int[]> anchors, Map<String, Object> streamer) {
            List<Map<String, Object>> free = new ArrayList<>();
            for (Map<String, Object> host : order) if (!anchors.containsKey(host) && host != streamer) free.add(host);
            int cursor = 0;
            List<Map<String, Object>> groups = new ArrayList<>();
            for (int next = 0; next <= anchored.size(); next++) {
                Map<String, Object> host = next < anchored.size() ? anchored.get(next) : null;
                int boundary = host == null ? pieces.size() : anchors.get(host)[0];
                for (Iterator<Map<String, Object>> it = free.iterator(); it.hasNext() && cursor < boundary;) {
                    Map<String, Object> filler = it.next();
                    int end = longest(filler, cursor, Collections.emptyList(), boundary);
                    if (end > cursor) {
                        groups.add(group(cursor, end, filler, "resident"));
                        cursor = end;
                        it.remove();
                    }
                }
                if (host == null) break;
                if (host == streamer) {
                    if (next != anchored.size() - 1) return null;
                    List<Map<String, Object>> planned = streaming(host, cursor);
                    if (planned == null) return null;
                    groups.addAll(planned);
                    return groups;
                }
                int limit = next + 1 < anchored.size() ? anchors.get(anchored.get(next + 1))[0] : pieces.size();
                int end = longest(host, cursor, Collections.emptyList(), limit);
                if (end > cursor) {
                    groups.add(group(cursor, end, host, "resident"));
                    cursor = end;
                }
            }
            if (cursor < pieces.size() && streamer != null) {
                List<Map<String, Object>> planned = streaming(streamer, cursor);
                if (planned == null) return null;
                groups.addAll(planned);
                cursor = pieces.size();
            }
            return cursor == pieces.size() ? groups : null;
        }
    }

    private List<Map<String, Object>> allocate(List<Map<String, Object>> stages, List<Map<String, Object>> hosts,
            CapacityStore.State state, String mode, int start, int device, Set<String> impossible) {
        if (start == stages.size()) return new ArrayList<>();
        if (device == hosts.size()) return null;
        String key = start + ":" + device;
        if (!impossible.add(key)) return null;
        Map<String, Object> host = hosts.get(device);
        List<Map<String, Object>> prefix = new ArrayList<>();
        int end = start;
        while (end < stages.size()) {
            Map<String, Object> stage = new LinkedHashMap<>(stages.get(end));
            stage.put("hostId", host.get("hostId"));
            stage.put("bootId", host.get("bootId"));
            stage.put("weights", mode);
            Map<String, Object> graph = new LinkedHashMap<>(graph(stage));
            graph.put("weights", mode);
            stage.put("graph", graph);
            prefix.add(stage);
            if (!fits(host, prefix, state)) { prefix.remove(prefix.size() - 1); break; }
            end++;
        }
        for (int length = prefix.size(); length > 0; length--) {
            List<Map<String, Object>> suffix = allocate(stages, hosts, state, mode, start + length, device + 1, impossible);
            if (suffix != null) {
                List<Map<String, Object>> result = new ArrayList<>(prefix.subList(0, length));
                result.addAll(suffix);
                return result;
            }
        }
        return allocate(stages, hosts, state, mode, start, device + 1, impossible);
    }

    private static Set<String> graphFiles(Object value) {
        Set<String> result = new HashSet<>();
        if (value instanceof Map) {
            Map<?, ?> object = (Map<?, ?>) value;
            if (object.get("file") instanceof String) result.add((String) object.get("file"));
            for (Object child : object.values()) result.addAll(graphFiles(child));
        } else if (value instanceof List) {
            for (Object child : (List<?>) value) result.addAll(graphFiles(child));
        }
        return result;
    }

    private boolean fits(Map<String, Object> host, List<Map<String, Object>> proposed, CapacityStore.State state) {
        List<Map<String, Object>> all = new ArrayList<>();
        for (Map<String, Object> plan : state.plans.values()) if ("ACTIVE".equals(plan.get("state"))) {
            for (Map<String, Object> stage : stages(plan)) if (host.get("hostId").equals(stage.get("hostId"))) all.add(stage);
        }
        all.addAll(proposed);
        long maxAllocation = 0;
        Map<String, Map<String, Object>> files = new HashMap<>();
        for (Map<String, Object> stage : all) {
            maxAllocation = Math.max(maxAllocation, optional(stage, "maxAllocationBytes"));
            for (Map<String, Object> file : objects(stage, "files", 1 << 16)) files.put((String) file.get("path"), file);
        }
        int sessionLimit = host.get("capacitySessionLimit") != null
                ? (int) Math.min(Integer.MAX_VALUE, number(host, "capacitySessionLimit"))
                : host.get("sessionLimit") == null ? 8 : (int) Math.min(Integer.MAX_VALUE, number(host, "sessionLimit"));
        if (host.get("affinityLimit") != null) sessionLimit = (int) Math.min(sessionLimit, number(host, "affinityLimit"));
        long active = optional(host, "activeSessions");
        long knownActive = 0;
        for (Map<String, Object> plan : state.plans.values()) if ("ACTIVE".equals(plan.get("state"))) {
            for (Map<String, Object> stage : stages(plan)) {
                if (host.get("hostId").equals(stage.get("hostId")) && Boolean.TRUE.equals(stage.get("opened"))) knownActive++;
            }
        }
        // Legacy sessions have no peak reservation: wait for them rather than reclaim their allocations.
        if (active > knownActive || all.size() > sessionLimit) return false;
        long[] need = required(host, all);
        if (host.get("gpuMaxAllocationBytes") != null && maxAllocation > number(host, "gpuMaxAllocationBytes")) return false;
        if (need[0] > gpuBudget(host) || need[1] > ramBudget(host)) return false;
        Map<String, Map<String, Object>> cached = cachedFiles(host);
        long disk = HEADROOM;
        for (Map<String, Object> file : files.values()) {
            if (!isCached(cached, file)) disk = Math.addExact(disk, number(file, "bytes"));
        }
        return disk <= number(host, "diskAvailableBytes");
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> stages(Map<String, Object> plan) {
        return (List<Map<String, Object>>) plan.get("stages");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> graph(Map<String, Object> stage) {
        return (Map<String, Object>) stage.get("graph");
    }

    public Future<String> pin(String cluster, String planId, List<String> bindings,
                              Map<String, Object> payload, List<String> files) {
        return store.atomic(cluster, state -> {
            Map<String, Object> plan = state.plans.get(planId);
            if (plan == null || !"ACTIVE".equals(plan.get("state"))) throw new IllegalArgumentException("Capacity plan unavailable");
            String host = null;
            Set<String> allowed = new HashSet<>();
            for (String binding : bindings) {
                Map<String, Object> found = null;
                for (Map<String, Object> stage : stages(plan)) if (binding.equals(stage.get("binding"))) found = stage;
                if (found == null) throw new IllegalArgumentException("Session is not in this capacity plan");
                if (Boolean.TRUE.equals(found.get("closed"))) throw new IllegalArgumentException("Planned session is closed");
                if (host != null && !host.equals(found.get("hostId"))) throw new IllegalArgumentException("Planned chain crosses device boundaries");
                host = (String) found.get("hostId");
                Map<String, Object> snapshot = state.hosts.get(host);
                if (snapshot == null || !found.get("bootId").equals(snapshot.get("bootId"))) {
                    throw new IllegalStateException("Planned worker restarted; start a new inference");
                }
                for (Map<String, Object> file : objects(found, "files", 1 << 16)) allowed.add((String) file.get("path"));
                Object actualGraph = payload.get("graph");
                if ("chain".equals(payload.get("op"))) {
                    for (Map<String, Object> stage : objects(payload, "stages", 1024)) {
                        if (NativeSessionKey.session(cluster, (String) found.get("session")).equals(stage.get("session"))) {
                            actualGraph = stage.get("graph");
                        }
                    }
                }
                if (actualGraph != null && !mapper.writeValueAsString(actualGraph).equals(mapper.writeValueAsString(graph(found)))) {
                    throw new IllegalArgumentException("Graph differs from pinned capacity plan");
                }
                if (!"close".equals(payload.get("op")) && !Boolean.TRUE.equals(found.get("opened")) && actualGraph == null) {
                    throw new IllegalArgumentException("First planned step requires its graph");
                }
            }
            if (!allowed.containsAll(files)) throw new IllegalArgumentException("Files differ from pinned capacity plan");
            return host;
        });
    }

    public Future<Void> completed(String cluster, String planId, List<String> bindings, boolean closed) {
        return store.atomic(cluster, state -> {
            Map<String, Object> plan = state.plans.get(planId);
            if (plan == null || !"ACTIVE".equals(plan.get("state"))) throw new IllegalStateException("Capacity plan unavailable");
            for (Map<String, Object> stage : stages(plan)) if (bindings.contains(stage.get("binding"))) {
                stage.put(closed ? "closed" : "opened", true);
            }
            return null;
        });
    }

    public Future<List<String>> bindings(String cluster, String planId) {
        return store.atomic(cluster, state -> {
            Map<String, Object> plan = state.plans.get(planId);
            if (plan == null) throw new IllegalArgumentException("Capacity plan unavailable");
            List<String> result = new ArrayList<>();
            for (Map<String, Object> stage : stages(plan)) result.add((String) stage.get("binding"));
            return result;
        });
    }

    public Future<Void> release(String cluster, String planId) {
        return store.atomic(cluster, state -> {
            Map<String, Object> plan = state.plans.get(planId);
            if (plan == null) throw new IllegalArgumentException("Capacity plan unavailable");
            for (Map<String, Object> stage : stages(plan)) {
                if (!Boolean.TRUE.equals(stage.get("closed"))) throw new IllegalStateException("Close every planned session before releasing capacity");
            }
            plan.put("state", "RELEASED");
            plan.put("releasedAt", System.currentTimeMillis());
            return null;
        });
    }

    private static Map<String, Object> map(Object... pairs) {
        Map<String, Object> result = new LinkedHashMap<>();
        for (int i = 0; i < pairs.length; i += 2) result.put((String) pairs[i], pairs[i + 1]);
        return result;
    }
}

package in.ramanujan.orchestrator.data.dao;

import in.ramanujan.db.layer.utils.CapacityStore;
import in.ramanujan.db.layer.utils.QueryExecutor;
import io.vertx.core.Future;
import org.junit.Test;

import java.io.RandomAccessFile;
import java.lang.reflect.Field;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;

import static org.junit.Assert.*;

public class CapacityDaoTest {
    private static final long MB = 1024L * 1024;
    private static final long GB = 1024L * MB;

    private static Map<String, Object> map(Object... pairs) {
        Map<String, Object> result = new LinkedHashMap<>();
        for (int i = 0; i < pairs.length; i += 2) result.put((String) pairs[i], pairs[i + 1]);
        return result;
    }

    private static void inject(Object target, String name, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(name);
        field.setAccessible(true);
        field.set(target, value);
    }

    private static <T> T await(Future<T> future) throws Exception {
        CompletableFuture<T> value = new CompletableFuture<>();
        future.setHandler(result -> {
            if (result.succeeded()) value.complete(result.result());
            else value.completeExceptionally(result.cause());
        });
        return value.get(5, TimeUnit.SECONDS);
    }

    private static void rejected(Future<?> future, String message) throws Exception {
        try { await(future); fail("Expected rejection"); }
        catch (ExecutionException expected) { assertTrue(expected.getCause().getMessage(), expected.getCause().getMessage().contains(message)); }
    }

    private static class Fixture implements AutoCloseable {
        final CapacityStore store = new CapacityStore();
        final CapacityDao dao = new CapacityDao();
        final List<Path> files = new ArrayList<>();

        Fixture() throws Exception {
            QueryExecutor query = new QueryExecutor();
            query.init(null, QueryExecutor.DB_TYPE.IN_MEM, null);
            inject(store, "queryExecutor", query);
            inject(dao, "store", store);
        }

        Map<String, Object> stats(String host, long disk) {
            return map("clusterId", "room", "hostId", host, "bootId", "boot-1", "sequence", 1L, "schemaVersion", 1,
                    "ramTotalBytes", 2 * GB, "ramAvailableBytes", 1536 * MB, "gpuTotalBytes", 2 * GB,
                    "diskAvailableBytes", disk, "unifiedMemory", false, "acceptingNewWork", true,
                    "sessionLimit", 16, "activeSessions", 0, "nativeReady", true);
        }

        Map<String, Object> stage(String id, long weights, long state) throws Exception {
            Path file = Files.createTempFile(Paths.get("."), "capacity_", ".bin").toAbsolutePath();
            files.add(file);
            try (RandomAccessFile sparse = new RandomAccessFile(file.toFile(), "rw")) { sparse.setLength(weights); }
            return map("affinity", id, "session", id, "weightBytes", weights, "stateBytes", state,
                    "streamWorkingBytes", 128 * MB, "scratchBytes", 32 * MB,
                    "graph", map("weights", "auto", "maxContext", 256),
                    "files", Arrays.asList(map("path", file.toString(), "bytes", weights)));
        }

        Map<String, Object> request(String mode, Map<String, Object>... stages) {
            return map("clusterId", "room", "weights", mode, "stages", Arrays.asList(stages));
        }

        @Override public void close() throws Exception { for (Path file : files) Files.deleteIfExists(file); }
    }

    @Test
    @SuppressWarnings("unchecked")
    public void oversizedModelStreamsMultipleContiguousShardsPerDevice() throws Exception {
        try (Fixture f = new Fixture()) {
            await(f.dao.report(f.stats("a", 9 * GB)));
            await(f.dao.report(f.stats("b", 9 * GB)));
            Map<String, Object> result = await(f.dao.reserve(f.request("auto",
                    f.stage("0", 4 * GB, 32 * MB), f.stage("1", 4 * GB, 32 * MB),
                    f.stage("2", 4 * GB, 32 * MB), f.stage("3", 4 * GB, 32 * MB))));
            List<Map<String, Object>> stages = (List<Map<String, Object>>) result.get("stages");
            assertEquals(Arrays.asList("a", "a", "b", "b"), Arrays.asList(
                    stages.get(0).get("hostId"), stages.get(1).get("hostId"), stages.get(2).get("hostId"), stages.get(3).get("hostId")));
            for (Map<String, Object> stage : stages) assertEquals("stream", stage.get("weights"));
            // 16 GiB of actual sparse weight files exceed the cluster's 8 GiB combined RAM + VRAM.
            assertEquals(4, stages.size());
        }
    }

    @Test
    public void oneDeviceMayOwnAllShardsButCannotOvercommitConcurrentState() throws Exception {
        try (Fixture f = new Fixture()) {
            await(f.dao.report(f.stats("a", 32 * GB)));
            Map<String, Object> first = f.stage("one", 4 * GB, 400 * MB);
            Map<String, Object> plan = await(f.dao.reserve(f.request("stream", first)));
            rejected(f.dao.reserve(f.request("stream", f.stage("two", 4 * GB, 400 * MB))), "lack fresh capacity");
            rejected(f.dao.release("room", (String) plan.get("planId")), "Close every");
            await(f.dao.completed("room", (String) plan.get("planId"), await(f.dao.bindings("room", (String) plan.get("planId"))), true));
            await(f.dao.release("room", (String) plan.get("planId")));
            assertNotNull(await(f.dao.reserve(f.request("stream", f.stage("three", 4 * GB, 400 * MB)))).get("planId"));
        }
    }

    @Test
    public void sequenceBootAndFreshnessCannotReassignAnActivePlan() throws Exception {
        try (Fixture f = new Fixture()) {
            Map<String, Object> stats = f.stats("a", 16 * GB);
            assertEquals(true, await(f.dao.report(stats)).get("accepted"));
            assertEquals(false, await(f.dao.report(stats)).get("accepted"));
            Map<String, Object> stage = f.stage("one", 4 * GB, 32 * MB);
            Map<String, Object> plan = await(f.dao.reserve(f.request("stream", stage)));
            String planId = (String) plan.get("planId");
            List<String> bindings = await(f.dao.bindings("room", planId));
            await(f.store.atomic("room", state -> { state.hosts.get("a").put("receivedAt", System.currentTimeMillis() - 46_000); return null; }));
            rejected(f.dao.reserve(f.request("stream", f.stage("two", 4 * GB, 32 * MB))), "lack fresh capacity");
            Map<String, Object> graph = map("weights", "stream", "maxContext", 256);
            assertEquals("a", await(f.dao.pin("room", planId, bindings, map("op", "step", "graph", graph), Collections.emptyList())));
            await(f.dao.report(mapWith(stats, "bootId", "boot-2")));
            assertEquals(false, await(f.dao.report(mapWith(stats, "sequence", 100))).get("accepted"));
            rejected(f.dao.pin("room", planId, bindings, map("op", "step", "graph", graph), Collections.emptyList()), "restarted");
        }
    }

    @Test
    public void sharedMemoryUsesPhysicalRamHeadroomAndUnknownRamIsNotUnlimited() throws Exception {
        try (Fixture f = new Fixture()) {
            Map<String, Object> stats = f.stats("a", 16 * GB);
            stats.put("unifiedMemory", true);
            stats.put("ramAvailableBytes", 700 * MB);
            stats.put("gpuTotalBytes", 16 * GB);
            await(f.dao.report(stats));
            rejected(f.dao.reserve(f.request("stream", f.stage("shared", 4 * GB, 600 * MB))), "lack fresh capacity");
            stats.put("sequence", 2);
            stats.put("ramAvailableBytes", null);
            await(f.dao.report(stats));
            rejected(f.dao.reserve(f.request("stream", f.stage("unknown", 4 * GB, 1))), "lack fresh capacity");
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    public void matchingCachedArtifactsAvoidDoubleChargingDiskAndGraphsArePinned() throws Exception {
        try (Fixture f = new Fixture()) {
            Map<String, Object> stage = f.stage("cached", 4 * GB, 32 * MB);
            Map<String, Object> file = ((List<Map<String, Object>>) stage.get("files")).get(0);
            file.put("mtime", Files.getLastModifiedTime(Paths.get((String) file.get("path"))).toMillis());
            Map<String, Object> stats = f.stats("a", 512 * MB);
            stats.put("cachedFiles", Arrays.asList(file));
            await(f.dao.report(stats));
            Map<String, Object> plan = await(f.dao.reserve(f.request("stream", stage)));
            String id = (String) plan.get("planId");
            rejected(f.dao.pin("room", id, await(f.dao.bindings("room", id)),
                    map("op", "step", "graph", map("weights", "resident", "maxContext", 256)), Collections.emptyList()), "Graph differs");
            rejected(f.dao.pin("other-room", id, Collections.emptyList(), map("op", "close"), Collections.emptyList()), "unavailable");
        }
    }

    @Test
    public void reservationsAreAtomicAndFailedAdmissionRollsBack() throws Exception {
        try (Fixture f = new Fixture()) {
            await(f.dao.report(f.stats("a", 16 * GB)));
            Map<String, Object> first = f.request("stream", f.stage("first", 4 * GB, 400 * MB));
            Map<String, Object> second = f.request("stream", f.stage("second", 4 * GB, 400 * MB));
            ExecutorService threads = Executors.newFixedThreadPool(2);
            try {
                CountDownLatch start = new CountDownLatch(1);
                List<java.util.concurrent.Future<Boolean>> results = new ArrayList<>();
                for (Map<String, Object> request : Arrays.asList(first, second)) results.add(threads.submit(() -> {
                    start.await();
                    return f.dao.reserve(request).succeeded();
                }));
                start.countDown();
                int accepted = 0;
                for (java.util.concurrent.Future<Boolean> result : results) if (result.get(5, TimeUnit.SECONDS)) accepted++;
                assertEquals(1, accepted);
                assertEquals(1, (int) await(f.store.atomic("room", state -> state.plans.size())));
            } finally { threads.shutdownNow(); }
        }
    }

    private static Map<String, Object> device(Fixture f, String id, long gpu) {
        Map<String, Object> stats = f.stats(id, 64 * GB);
        stats.put("gpuTotalBytes", gpu);
        stats.put("ramTotalBytes", 16 * GB);
        stats.put("ramAvailableBytes", 12 * GB);
        return stats;
    }

    private static Map<String, Object> vram(Fixture f, int count, String mode, boolean dryRun) throws Exception {
        List<Map<String, Object>> pieces = new ArrayList<>();
        for (int index = 0; index < count; index++) {
            Map<String, Object> piece = f.stage("p" + index, GB, MB);
            Map<String, Object> graph = map("weights", "auto", "maxContext", 256,
                    "layers", Collections.singletonList(map("index", index)));
            if (index == 0) graph.put("embed", map("rows", 1));
            if (index == count - 1) graph.put("head", map("rows", 1));
            piece.put("graph", graph);
            piece.put("sharedScratchBytes", 32 * MB);
            piece.put("streamItemBytes", 128 * MB);
            piece.put("streamSharedBytes", 0);
            pieces.add(piece);
        }
        Map<String, Object> request = map("clusterId", "room", "weights", mode, "placement", "vram", "stages", pieces);
        if (dryRun) request.put("dryRun", true);
        return request;
    }

    @Test
    @SuppressWarnings("unchecked")
    public void orchestratorShardsBigToSmallByVramAndMergesPieces() throws Exception {
        try (Fixture f = new Fixture()) {
            await(f.dao.report(device(f, "a-small", 8 * GB)));
            await(f.dao.report(device(f, "z-big", 16 * GB)));
            Map<String, Object> snapshot = null;
            for (Map<String, Object> host : await(f.dao.snapshots("room"))) if ("z-big".equals(host.get("hostId"))) snapshot = host;
            assertEquals(16 * GB - (16 * GB / 10) - 256 * MB, snapshot.get("gpuBudgetBytes"));
            Map<String, Object> dry = await(f.dao.reserve(vram(f, 20, "auto", true)));
            assertNull(dry.get("planId"));
            assertEquals(0, (int) await(f.store.atomic("room", state -> state.plans.size())));
            Map<String, Object> plan = await(f.dao.reserve(vram(f, 20, "auto", false)));
            List<Map<String, Object>> stages = (List<Map<String, Object>>) plan.get("stages");
            assertEquals(dry.get("stages").toString().length(), stages.toString().length());
            assertEquals(2, stages.size());
            // 16 GiB device: 14.15 GiB usable -> 14 one-GiB layers; the 8 GiB device takes the other 6.
            assertEquals("z-big", stages.get(0).get("hostId"));
            assertEquals(Arrays.asList(0, 14), stages.get(0).get("pieces"));
            assertEquals("a-small", stages.get(1).get("hostId"));
            assertEquals(Arrays.asList(14, 20), stages.get(1).get("pieces"));
            for (Map<String, Object> stage : stages) assertEquals("resident", stage.get("weights"));
            Map<String, Object> graph = (Map<String, Object>) stages.get(0).get("graph");
            assertEquals(14, ((List<?>) graph.get("layers")).size());
            assertTrue(graph.containsKey("embed"));
            assertFalse(graph.containsKey("head"));
            assertEquals("resident", graph.get("weights"));
            assertTrue(((Map<String, Object>) stages.get(1).get("graph")).containsKey("head"));
            assertEquals("p0", stages.get(0).get("affinity"));
            assertEquals("p14", stages.get(1).get("session"));
            String id = (String) plan.get("planId");
            assertEquals("z-big", await(f.dao.pin("room", id, Collections.singletonList((String) await(f.dao.bindings("room", id)).get(0)),
                    map("op", "step", "graph", graph), Collections.emptyList())));
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    public void modelThatFitsOneDeviceIsNotSplitAndOversizedModelsStreamTheLeast() throws Exception {
        try (Fixture f = new Fixture()) {
            await(f.dao.report(device(f, "small", 8 * GB)));
            await(f.dao.report(device(f, "big", 16 * GB)));
            List<Map<String, Object>> single = (List<Map<String, Object>>) await(f.dao.reserve(vram(f, 6, "auto", true))).get("stages");
            assertEquals(1, single.size());
            assertEquals("big", single.get(0).get("hostId"));
            List<Map<String, Object>> stages = (List<Map<String, Object>>) await(f.dao.reserve(vram(f, 24, "auto", true))).get("stages");
            long resident = 0, streamed = 0, big = 0, small = 0;
            int expected = 0;
            Set<String> finished = new HashSet<>();
            String previous = null;
            for (Map<String, Object> stage : stages) {
                List<Integer> range = (List<Integer>) stage.get("pieces");
                assertEquals(expected, (int) range.get(0));
                expected = range.get(1);
                String host = (String) stage.get("hostId");
                if (!host.equals(previous)) { assertTrue(finished.add(host)); previous = host; }
                long bytes = ((Number) stage.get("weightBytes")).longValue();
                if ("stream".equals(stage.get("weights"))) streamed += bytes;
                else {
                    resident += bytes;
                    if ("big".equals(host)) big += bytes; else small += bytes;
                }
            }
            assertEquals(24, expected);
            assertTrue(streamed > 0);
            assertEquals(24 * GB, resident + streamed);
            assertTrue(big > small);
            // The streamed range never includes the output head when the head alone fits resident.
            assertEquals("resident", stages.get(stages.size() - 1).get("weights"));
            rejected(f.dao.reserve(vram(f, 24, "resident", true)), "lack fresh capacity");
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    public void vramPlacementValidatesRequests() throws Exception {
        try (Fixture f = new Fixture()) {
            await(f.dao.report(device(f, "big", 16 * GB)));
            rejected(f.dao.reserve(vram(f, 2, "stream", false)), "resident");
            Map<String, Object> search = f.request("auto", f.stage("s", MB, MB));
            search.put("dryRun", true);
            rejected(f.dao.reserve(search), "dryRun");
            Map<String, Object> missing = vram(f, 2, "auto", false);
            ((List<Map<String, Object>>) missing.get("stages")).get(1).remove("streamItemBytes");
            rejected(f.dao.reserve(missing), "streamItemBytes");
            Map<String, Object> inconsistent = vram(f, 2, "auto", false);
            ((List<Map<String, Object>>) inconsistent.get("stages")).get(0).put("streamSharedBytes", 1);
            rejected(f.dao.reserve(inconsistent), "inconsistent");
        }
    }

    /** Ping fields for a device that cached {@code [start, end)} of {@code request}'s pieces. */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> cachedDevice(Fixture f, String id, long gpu, Map<String, Object> request,
            int start, int end, String model, long mtimeShift) throws Exception {
        Map<String, Object> stats = device(f, id, gpu);
        List<Map<String, Object>> files = new ArrayList<>();
        List<Map<String, Object>> pieces = (List<Map<String, Object>>) request.get("stages");
        for (int index = start; index < end; index++) {
            for (Map<String, Object> file : (List<Map<String, Object>>) pieces.get(index).get("files")) {
                long mtime = Files.getLastModifiedTime(Paths.get((String) file.get("path"))).toMillis() + mtimeShift;
                files.add(map("path", file.get("path"), "bytes", file.get("bytes"), "mtime", mtime));
            }
        }
        stats.put("cachedFiles", files);
        stats.put("cachedShards", Collections.singletonList(map("model", model, "embed", start == 0,
                "head", end == pieces.size(), "layers", Collections.singletonList(Arrays.asList(start, end)),
                "bytes", (end - start) * GB)));
        return stats;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> modelRequest(Fixture f, int count) throws Exception {
        Map<String, Object> request = vram(f, count, "auto", true);
        for (Map<String, Object> piece : (List<Map<String, Object>>) request.get("stages")) {
            ((Map<String, Object>) piece.get("graph")).put("hyper", map("dim", 8, "eps", 1.0E-6));
        }
        return request;
    }

    @Test
    public void modelKeyIsTheSortedHyperDigestWorkersUse() throws Exception {
        // Same literal as WorkerShardManifestTest: both sides must name a model identically.
        assertEquals(MODEL_KEY, new CapacityDao().modelKey(map("rope_theta", 1.0E7, "dim", 5120, "eps", 1.0E-6,
                "ssm", map("d_state", 128, "conv", 4), "architecture", "qwen35")));
    }

    static final String MODEL_KEY = "878d4985f9a5823f297ba33895d9689943409f70b93d4f5492e1fb8ce2cb80e2";

    @Test
    @SuppressWarnings("unchecked")
    public void rejoiningDeviceKeepsItsCachedShardAndOnlyTheRemainderIsSharded() throws Exception {
        try (Fixture f = new Fixture()) {
            Map<String, Object> request = modelRequest(f, 20);
            String model = f.dao.modelKey(map("dim", 8, "eps", 1.0E-6));
            // Without the cache, the 16 GiB device takes [0,14) and the 8 GiB device [14,20).
            await(f.dao.report(device(f, "z-big", 16 * GB)));
            await(f.dao.report(cachedDevice(f, "a-small", 8 * GB, request, 0, 6, model, 0)));
            List<Map<String, Object>> stages = (List<Map<String, Object>>) await(f.dao.reserve(request)).get("stages");
            assertEquals(2, stages.size());
            assertEquals("a-small", stages.get(0).get("hostId"));
            assertEquals(Arrays.asList(0, 6), stages.get(0).get("pieces"));
            assertEquals(0L, ((Number) stages.get(0).get("downloadBytes")).longValue());
            assertEquals("z-big", stages.get(1).get("hostId"));
            assertEquals(Arrays.asList(6, 20), stages.get(1).get("pieces"));
            assertEquals(14 * GB, ((Number) stages.get(1).get("downloadBytes")).longValue());
            for (Map<String, Object> stage : stages) assertEquals("resident", stage.get("weights"));

            // A cached middle range is kept too: the biggest free device fills the gap before it
            // and the next one the rest, so each device still holds one contiguous range.
            await(f.dao.report(device(f, "c-mid", 8 * GB)));
            Map<String, Object> middle = cachedDevice(f, "a-small", 8 * GB, request, 8, 13, model, 0);
            middle.put("sequence", 2L);
            await(f.dao.report(middle));
            stages = (List<Map<String, Object>>) await(f.dao.reserve(request)).get("stages");
            assertEquals(3, stages.size());
            assertEquals("z-big", stages.get(0).get("hostId"));
            assertEquals(Arrays.asList(0, 8), stages.get(0).get("pieces"));
            assertEquals("a-small", stages.get(1).get("hostId"));
            assertEquals(Arrays.asList(8, 14), stages.get(1).get("pieces"));
            assertEquals(GB, ((Number) stages.get(1).get("downloadBytes")).longValue());
            assertEquals("c-mid", stages.get(2).get("hostId"));
            assertEquals(Arrays.asList(14, 20), stages.get(2).get("pieces"));

            // Changed files or another model's shard are not trusted: back to plain big-to-small.
            for (Object[] variant : new Object[][]{{model, 1L}, {"other-model", 0L}}) {
                Map<String, Object> stale = cachedDevice(f, "a-small", 8 * GB, request, 0, 6, (String) variant[0], (Long) variant[1]);
                stale.put("sequence", variant[1].equals(1L) ? 3L : 4L);
                await(f.dao.report(stale));
                stages = (List<Map<String, Object>>) await(f.dao.reserve(request)).get("stages");
                assertEquals("z-big", stages.get(0).get("hostId"));
                assertEquals(Arrays.asList(0, 14), stages.get(0).get("pieces"));
            }
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    public void cachedShardReportsAreValidated() throws Exception {
        try (Fixture f = new Fixture()) {
            Map<String, Object> bad = device(f, "a", 8 * GB);
            bad.put("cachedShards", Collections.singletonList(map("model", "m", "embed", true, "head", false,
                    "layers", Collections.singletonList(Arrays.asList(5, 5)), "bytes", 1)));
            rejected(f.dao.report(bad), "empty");
            bad.put("cachedShards", Collections.singletonList(map("model", "m", "embed", true, "head", false,
                    "layers", Collections.singletonList(Collections.singletonList(1)), "bytes", 1)));
            rejected(f.dao.report(bad), "[start, end)");
        }
    }

    private static Map<String, Object> mapWith(Map<String, Object> input, String key, Object value) {
        Map<String, Object> result = new LinkedHashMap<>(input);
        result.put(key, value);
        return result;
    }
}

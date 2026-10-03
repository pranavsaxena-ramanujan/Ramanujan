package in.ramanujan.developer.console.operationImpl;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import in.ramanujan.pojo.RuleEngineInput;
import in.ramanujan.pojo.ruleEngineInputUnitsExt.array.Array;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.FileTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class WorkerBinaryCacheTest {
    private HttpServer server;
    private Path root;
    private final AtomicInteger fetches = new AtomicInteger();
    private volatile boolean truncate;

    @Before
    public void start() throws IOException {
        root = Files.createTempDirectory("worker-cache-test-");
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/binary/stat", ex -> {
            Path file = requested(ex);
            reply(ex, 200, ("{\"status\":\"SUCCESS\",\"size\":" + Files.size(file)
                    + ",\"mtime\":" + Files.getLastModifiedTime(file).toMillis() + "}").getBytes(StandardCharsets.UTF_8));
        });
        server.createContext("/binary/fetch", ex -> {
            fetches.incrementAndGet();
            byte[] bytes = Files.readAllBytes(requested(ex));
            if (truncate) {
                ex.sendResponseHeaders(200, bytes.length);
                try (OutputStream out = ex.getResponseBody()) {
                    out.write(bytes, 0, bytes.length / 2);
                }
                return;
            }
            reply(ex, 200, bytes);
        });
        server.start();
    }

    @After
    public void stop() throws IOException {
        server.stop(0);
        try (java.util.stream.Stream<Path> walk = Files.walk(root)) {
            walk.sorted(java.util.Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
        }
    }

    private static Path requested(HttpExchange ex) throws IOException {
        String query = ex.getRequestURI().getRawQuery();
        return Paths.get(URLDecoder.decode(query.substring("path=".length()), "UTF-8"));
    }

    private static void reply(HttpExchange ex, int code, byte[] body) throws IOException {
        ex.sendResponseHeaders(code, body.length);
        try (OutputStream out = ex.getResponseBody()) {
            out.write(body);
        }
    }

    private WorkerBinaryCache cache() {
        return new WorkerBinaryCache(root.resolve("cache"), "http://127.0.0.1:" + server.getAddress().getPort());
    }

    private static RuleEngineInput input(String... files) {
        List<Array> arrays = new ArrayList<>();
        for (String file : files) {
            Array array = new Array();
            array.setBinaryFile(file);
            arrays.add(array);
        }
        arrays.add(new Array());
        RuleEngineInput rei = new RuleEngineInput();
        rei.setArrays(arrays);
        return rei;
    }

    @Test
    public void differentPrivateGatewaysDoNotShareEqualPathAndStatCacheEntries() throws IOException {
        server.createContext("/worker/room-a/binary/stat",
                ex -> reply(ex, 200, "{\"size\":1,\"mtime\":1}".getBytes(StandardCharsets.UTF_8)));
        server.createContext("/worker/room-b/binary/stat",
                ex -> reply(ex, 200, "{\"size\":1,\"mtime\":1}".getBytes(StandardCharsets.UTF_8)));
        server.createContext("/worker/room-a/binary/fetch", ex -> reply(ex, 200, new byte[]{1}));
        server.createContext("/worker/room-b/binary/fetch", ex -> reply(ex, 200, new byte[]{2}));
        String gateway = "http://127.0.0.1:" + server.getAddress().getPort();
        WorkerBinaryCache a = new WorkerBinaryCache(root.resolve("cache"), gateway + "/worker/room-a");
        WorkerBinaryCache b = new WorkerBinaryCache(root.resolve("cache"), gateway + "/worker/room-b");
        try {
            Path first = a.cached("/models/shared-name.bin");
            Path second = b.cached("/models/shared-name.bin");
            assertNotEquals(first, second);
            assertArrayEquals(new byte[]{1}, Files.readAllBytes(first));
            assertArrayEquals(new byte[]{2}, Files.readAllBytes(second));
        } finally { a.close(); b.close(); }
    }

    @Test
    public void weightsAreDownloadedOnceAndMutableStateEveryTask() throws IOException {
        Path server = Files.createDirectories(root.resolve("server/layer 0"));
        Path weight = Files.write(server.resolve("attn_q.bin"), new byte[] {1, 2, 3, 4});
        Path state = Files.write(server.resolve("hidden.bin"), new byte[] {9, 9});

        WorkerBinaryCache cache = cache();
        RuleEngineInput first = input(weight.toString(), state.toString(), weight.toString());
        cache.localize(first, root.resolve("tasks/t1"));
        String localWeight = first.getArrays().get(0).getBinaryFile();
        String localState = first.getArrays().get(1).getBinaryFile();
        assertEquals(localWeight, first.getArrays().get(2).getBinaryFile());
        assertTrue(localWeight.endsWith("/attn_q.bin"));
        assertTrue("mutable name is preserved", localState.endsWith("/hidden.bin"));
        assertTrue(localState.startsWith(root.resolve("tasks/t1").toString()));
        assertArrayEquals(new byte[] {1, 2, 3, 4}, Files.readAllBytes(Paths.get(localWeight)));
        assertEquals(2, fetches.get());

        Files.write(state, new byte[] {7, 7});
        RuleEngineInput second = input(weight.toString(), state.toString());
        cache.localize(second, root.resolve("tasks/t2"));
        assertEquals(localWeight, second.getArrays().get(0).getBinaryFile());
        assertArrayEquals(new byte[] {7, 7}, Files.readAllBytes(Paths.get(second.getArrays().get(1).getBinaryFile())));
        assertEquals("only the mutable state is fetched again", 3, fetches.get());

        RuleEngineInput restarted = input(weight.toString());
        cache().localize(restarted, root.resolve("tasks/t3"));
        assertEquals("a restarted worker reuses its disk cache", 3, fetches.get());
    }

    @Test
    public void changedWeightIsFetchedAgainByNewWorker() throws IOException {
        Path weight = Files.write(Files.createDirectories(root.resolve("server")).resolve("w.bin"), new byte[] {1});
        cache().localize(input(weight.toString()), root.resolve("tasks/a"));
        Files.write(weight, new byte[] {2});
        Files.setLastModifiedTime(weight, FileTime.fromMillis(Files.getLastModifiedTime(weight).toMillis() + 5_000));
        RuleEngineInput rei = input(weight.toString());
        cache().localize(rei, root.resolve("tasks/b"));
        assertEquals(2, fetches.get());
        assertArrayEquals(new byte[] {2}, Files.readAllBytes(Paths.get(rei.getArrays().get(0).getBinaryFile())));
    }

    @Test
    public void truncatedTransferIsNeverCached() throws IOException {
        Path weight = Files.write(Files.createDirectories(root.resolve("server")).resolve("w.bin"), new byte[64]);
        truncate = true;
        try {
            cache().localize(input(weight.toString()), root.resolve("tasks/a"));
            fail("truncated fetch must fail");
        } catch (IOException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("received"));
        }
        truncate = false;
        RuleEngineInput rei = input(weight.toString());
        cache().localize(rei, root.resolve("tasks/b"));
        assertEquals(64, Files.size(Paths.get(rei.getArrays().get(0).getBinaryFile())));
        try (java.util.stream.Stream<Path> walk = Files.walk(root.resolve("cache"))) {
            assertFalse(walk.anyMatch(p -> p.toString().endsWith(".partial")));
        }
    }

    @Test
    public void cacheInventoryReportsOriginalFingerprintAndMarksBoundedResults() throws IOException {
        Path directory = Files.createDirectories(root.resolve("server"));
        Path a = Files.write(directory.resolve("a.bin"), new byte[8]);
        Path b = Files.write(directory.resolve("b.bin"), new byte[12]);
        WorkerBinaryCache cache = cache();
        cache.cached(a.toString());
        cache.cached(b.toString());
        WorkerBinaryCache.CacheInventory all = cache.inventory();
        assertEquals(2, all.files.size());
        assertFalse(all.truncated);
        for (java.util.Map<String, Object> entry : all.files) {
            Path backend = Paths.get((String) entry.get("path"));
            assertEquals(Files.size(backend), ((Number) entry.get("bytes")).longValue());
            assertEquals(Files.getLastModifiedTime(backend).toMillis(), ((Number) entry.get("mtime")).longValue());
        }
        WorkerBinaryCache.CacheInventory bounded = cache.inventory(1);
        assertEquals(1, bounded.files.size());
        assertTrue(bounded.truncated);
        WorkerCapacity snapshot = new WorkerCapacity(p -> new WorkerCapacity.Snapshot(null, null, null),
                root.resolve("cache"), "host", cache);
        assertEquals(all.files.size(), ((List<?>) snapshot.sample(0, true).get("cachedFiles")).size());
        cache.close();
        WorkerBinaryCache restarted = cache();
        try { assertEquals(2, restarted.inventory().files.size()); }
        finally { restarted.close(); }
    }

    @Test
    public void cacheInventoryExcludesOtherRoomsIncompleteFilesAndTokenPaths() throws IOException {
        server.createContext("/worker/room-a/binary/stat",
                ex -> reply(ex, 200, "{\"size\":1,\"mtime\":7}".getBytes(StandardCharsets.UTF_8)));
        server.createContext("/worker/room-b/binary/stat",
                ex -> reply(ex, 200, "{\"size\":1,\"mtime\":9}".getBytes(StandardCharsets.UTF_8)));
        server.createContext("/worker/room-a/binary/fetch", ex -> reply(ex, 200, new byte[]{1}));
        server.createContext("/worker/room-b/binary/fetch", ex -> reply(ex, 200, new byte[]{2}));
        String gateway = "http://127.0.0.1:" + server.getAddress().getPort();
        WorkerBinaryCache a = new WorkerBinaryCache(root.resolve("cache"), gateway + "/worker/room-a");
        WorkerBinaryCache b = new WorkerBinaryCache(root.resolve("cache"), gateway + "/worker/room-b");
        try {
            a.cached("/models/a.bin");
            Path bLocal = b.cached("/models/b.bin");
            a.cached("https://portal.example/worker/bearer-secret/weight.bin");
            a.cached("/worker/bearer-secret/weight.bin");
            assertEquals(1, a.inventory().files.size());
            assertEquals("/models/a.bin", a.inventory().files.get(0).get("path"));
            assertEquals(1, b.inventory().files.size());
            assertEquals("/models/b.bin", b.inventory().files.get(0).get("path"));
            Files.write(bLocal, new byte[2]);
            assertTrue(b.inventory().files.isEmpty());
        } finally { a.close(); b.close(); }
    }

    @Test
    public void assignedDiskBudgetIncludesAllUncachedWeightsBeforeAnyFetch() throws IOException {
        long size = Files.getFileStore(root).getUsableSpace() / 2 + (1L << 30);
        server.removeContext("/binary/stat");
        server.createContext("/binary/stat", ex ->
                reply(ex, 200, ("{\"size\":" + size + ",\"mtime\":1}").getBytes(StandardCharsets.UTF_8)));
        WorkerBinaryCache cache = cache();
        try {
            cache.preflight(Arrays.asList("/assigned/layer0.bin", "/assigned/layer1.bin"));
            fail("the full assigned immutable disk budget must be checked before downloads");
        } catch (IOException expected) {
            assertTrue(expected.getMessage().contains("Insufficient cache disk space"));
        } finally { cache.close(); }
        assertEquals(0, fetches.get());
    }

    @Test
    public void assignedDiskBudgetDeduplicatesAlreadyCachedWeightsAcrossRestart() throws IOException {
        Path weight = Files.write(Files.createDirectories(root.resolve("server")).resolve("w.bin"), new byte[8]);
        WorkerBinaryCache first = cache();
        first.cached(weight.toString());
        first.close();
        WorkerBinaryCache restarted = new WorkerBinaryCache(root.resolve("cache"),
                "http://127.0.0.1:" + server.getAddress().getPort(), Long.MAX_VALUE);
        try {
            restarted.preflight(Arrays.asList(weight.toString(), weight.toString()));
        } finally { restarted.close(); }
        assertEquals("cached weights have zero incremental download/disk budget", 1, fetches.get());
    }

    @Test
    public void diskReserveRejectsBeforeFetchingOrCreatingPartialFiles() throws IOException {
        Path weight = Files.write(Files.createDirectories(root.resolve("server")).resolve("w.bin"), new byte[8]);
        WorkerBinaryCache limited = new WorkerBinaryCache(root.resolve("cache"),
                "http://127.0.0.1:" + server.getAddress().getPort(), Long.MAX_VALUE);
        try {
            limited.cached(weight.toString());
            fail("expected disk reserve rejection");
        } catch (IOException expected) {
            assertTrue(expected.getMessage().contains("Insufficient cache disk space"));
        } finally { limited.close(); }
        assertEquals(0, fetches.get());
        try (java.util.stream.Stream<Path> walk = Files.walk(root.resolve("cache"))) {
            assertFalse(walk.anyMatch(p -> p.toString().endsWith(".partial")));
        }
    }

    @Test
    public void mutableNamesMatchNativeRuntime() {
        for (String name : Arrays.asList("hidden.bin", "layer_03_state.bin", "l1_k_cache.bin", "l1_v_cache.bin")) {
            assertTrue(name, WorkerBinaryCache.isMutable("/x/" + name));
        }
        assertFalse(WorkerBinaryCache.isMutable("/x/attn_k.bin"));
        assertNotEquals(true, WorkerBinaryCache.isMutable("/x/state.bin.bak"));
    }

    @Test
    public void copyBinaryAtomicallyLeavesSourceAndNoPartial() throws IOException {
        Path source = Files.write(root.resolve("src.bin"), new byte[] {5, 6});
        Path destination = root.resolve("dest.bin");
        Files.write(destination, new byte[] {0});
        ExecuteInlineHomelabServer.copyBinaryAtomically(source.toString(), destination.toString());
        assertArrayEquals(new byte[] {5, 6}, Files.readAllBytes(destination));
        assertTrue(Files.exists(source));
        assertFalse(Files.exists(root.resolve("dest.bin.partial")));
    }
}

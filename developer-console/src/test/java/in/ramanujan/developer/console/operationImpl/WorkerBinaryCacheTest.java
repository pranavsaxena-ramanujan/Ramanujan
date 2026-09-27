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

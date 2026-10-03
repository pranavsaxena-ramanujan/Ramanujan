package in.ramanujan.developer.console.operationImpl;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import in.ramanujan.rule.engine.LlmSession;
import org.junit.Assume;
import org.junit.Test;

import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.*;

/** Opt-in GPU smoke test: prepares runtime kernels only, never opens an inference session. */
public class WorkerCapacityNativeProbeTest {
    @Test public void firstHeartbeatHasSelectedGpuBeforeAnyInferencePlan() throws Exception {
        Assume.assumeTrue(Boolean.getBoolean("ramanujan.test.nativeProbe"));
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        ExecutorService http = Executors.newCachedThreadPool();
        server.setExecutor(http);
        CountDownLatch heartbeat = new CountDownLatch(1);
        AtomicReference<Map<?, ?>> snapshot = new AtomicReference<>();
        AtomicReference<Throwable> failure = new AtomicReference<>();
        server.createContext("/pings/open", exchange -> {
            byte[] body = "{\"status\":\"SUCCESS\",\"data\":null}".getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, body.length);
            try (java.io.OutputStream out = exchange.getResponseBody()) { out.write(body); }
        });
        server.createContext("/pings/capacity", exchange -> {
            snapshot.set(new ObjectMapper().readValue(exchange.getRequestBody(), Map.class));
            exchange.sendResponseHeaders(204, -1);
            exchange.close();
            heartbeat.countDown();
        });
        server.start();
        Path root = Paths.get("target", "capacity-native-" + UUID.randomUUID()).toAbsolutePath();
        ExecuteInlineWorker worker = new ExecuteInlineWorker();
        Thread running = new Thread(() -> {
            try {
                worker.execute(Arrays.asList("worker", "http://127.0.0.1:" + server.getAddress().getPort(),
                        "1", "--cache", root.toString()));
            } catch (Throwable e) { failure.set(e); }
        });
        try {
            running.start();
            assertTrue("first heartbeat must follow native preflight", heartbeat.await(30, TimeUnit.SECONDS));
            Map<?, ?> value = snapshot.get();
            assertEquals(true, value.get("nativeReady"));
            assertEquals(true, value.get("supportsRuntime"));
            assertEquals(true, value.get("supportsStreaming"));
            assertTrue(((Number) value.get("gpuTotalBytes")).longValue() > 0);
            assertTrue(((Number) value.get("gpuMaxAllocationBytes")).longValue() > 0);
            assertEquals(0L, ((Number) value.get("gpuAllocatedBytes")).longValue());
            assertEquals(0L, ((Number) value.get("gpuResidentBytes")).longValue());
            assertNull(value.get("gpuAvailableBytes"));
            assertNull(value.get("ramAllocatedBytes"));
            assertEquals(0, ((Number) value.get("activeSessions")).intValue());
            assertEquals(8, ((Number) value.get("capacitySessionLimit")).intValue());
            Map<?, ?> before = new ObjectMapper().readValue(LlmSession.capacityInfo(), Map.class);
            LlmSession.prepareCapacity();
            assertEquals("repeat probes must reuse selected state", before,
                    new ObjectMapper().readValue(LlmSession.capacityInfo(), Map.class));
        } finally {
            worker.stop();
            running.join(5000);
            server.stop(0);
            http.shutdownNow();
            Files.deleteIfExists(root.resolve("tasks"));
            Files.deleteIfExists(root);
        }
        assertFalse(running.isAlive());
        assertNull(failure.get());
    }
}

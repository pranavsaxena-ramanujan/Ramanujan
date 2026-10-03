package in.ramanujan.developer.console.operationImpl;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import in.ramanujan.devices.common.WorkerCapacity;
import org.junit.Test;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.*;

public class WorkerCapacityTest {
    @Test public void schemaUsesPhysicalProviderAndNullableUnknowns() throws Exception {
        Path root = Paths.get("target", "capacity-test-" + UUID.randomUUID());
        Files.createDirectories(root);
        try {
            Files.write(root.resolve("weight.bin"), new byte[17]);
            WorkerCapacity capacity = new WorkerCapacity(p -> new WorkerCapacity.Snapshot(100L, 40L, 70L), root, "host");
            Map<String, Object> first = capacity.sample(3, false, java.util.Collections.singletonMap("plan-a", 3));
            Map<String, Object> second = capacity.sample(3, true);
            assertEquals(29, first.size());
            assertEquals(1, first.get("schemaVersion"));
            assertEquals("host", first.get("hostId"));
            UUID.fromString((String) first.get("bootId"));
            assertEquals(first.get("bootId"), second.get("bootId"));
            assertEquals(1L, first.get("sequence"));
            assertEquals(2L, second.get("sequence"));
            assertEquals(100L, first.get("ramTotalBytes"));
            assertEquals(40L, first.get("ramAvailableBytes"));
            assertEquals(70L, first.get("diskAvailableBytes"));
            assertEquals(17L, first.get("cacheBytes"));
            assertNull(first.get("gpuAvailableBytes"));
            assertNull(first.get("ramAllocatedBytes"));
            assertNull(first.get("gpuAllocatedBytes"));
            assertNull(first.get("gpuMaxAllocationBytes"));
            assertNull(first.get("gpuDeviceName"));
            assertNull(first.get("gpuRuntime"));
            assertEquals(java.util.Collections.singletonMap("plan-a", 3), first.get("activePlans"));
            assertEquals(java.util.Collections.emptyList(), first.get("cachedFiles"));
            assertEquals(false, first.get("cachedFilesTruncated"));
            assertNull(first.get("nativeReady"));
            assertNull(first.get("supportsStreaming"));
            assertNull(first.get("supportsRuntime"));
            assertEquals(3, first.get("activeSessions"));
            assertEquals(false, first.get("acceptingNewWork"));
            assertEquals(true, second.get("acceptingNewWork"));
            Map<String, Object> unknown = new WorkerCapacity(p -> { throw new UnsupportedOperationException(); },
                    root, "host").sample(0, true);
            assertNull(unknown.get("ramTotalBytes"));
            assertNull(unknown.get("ramAvailableBytes"));
            assertNull(unknown.get("diskAvailableBytes"));
            assertNull(new WorkerCapacity.Snapshot(-1L, -2L, -3L).ramTotalBytes);
        } finally {
            Files.deleteIfExists(root.resolve("weight.bin"));
            Files.deleteIfExists(root);
        }
    }

    @Test public void desktopAvailableMemoryCountsReclaimableMacPages() {
        String text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                + "Pages free:                                     4934.\n"
                + "Pages active:                                  91022.\n"
                + "Pages inactive:                                90862.\n"
                + "Pages speculative:                               942.\n";
        assertEquals(Long.valueOf((4934L + 90862 + 942) * 16384),
                WorkerCapacity.DesktopProvider.parseVmStat(text));
        assertNull(WorkerCapacity.DesktopProvider.parseVmStat("Pages free: 1.\n"));
    }

    @Test public void diskGuardHandlesReserveConcurrencyAndOverflow() throws Exception {
        WorkerBinaryCache.checkDiskSpace(100, 70, 20, 10);
        for (long[] values : new long[][]{{100, 71, 20, 10}, {10, 0, 20, 0},
                {100, -1, 0, 0}, {Long.MAX_VALUE, Long.MAX_VALUE, 1, 0}}) {
            try {
                WorkerBinaryCache.checkDiskSpace(values[0], values[1], values[2], values[3]);
                fail("expected disk capacity error");
            } catch (IOException expected) { assertTrue(expected.getMessage().contains("Insufficient cache disk space")); }
        }
    }

    @Test public void heartbeatRunsWhilePollingIsBlockedAndUsesPollingIdentity() throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        ExecutorService http = Executors.newCachedThreadPool();
        server.setExecutor(http);
        CountDownLatch poll = new CountDownLatch(1), heartbeat = new CountDownLatch(1), release = new CountDownLatch(1);
        AtomicReference<String> query = new AtomicReference<>();
        AtomicReference<Map<?, ?>> payload = new AtomicReference<>();
        AtomicReference<Throwable> error = new AtomicReference<>();
        java.util.concurrent.atomic.AtomicInteger probes = new java.util.concurrent.atomic.AtomicInteger();
        AtomicReference<String> probeThread = new AtomicReference<>();
        server.createContext("/pings/open", exchange -> {
            query.set(exchange.getRequestURI().getQuery());
            poll.countDown();
            try { release.await(10, TimeUnit.SECONDS); }
            catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            exchange.sendResponseHeaders(204, -1);
            exchange.close();
        });
        server.createContext("/pings/capacity", exchange -> {
            payload.set(new ObjectMapper().readValue(exchange.getRequestBody(), Map.class));
            exchange.sendResponseHeaders(204, -1);
            exchange.close();
            heartbeat.countDown();
        });
        server.start();
        Path root = Paths.get("target", "capacity-worker-" + UUID.randomUUID()).toAbsolutePath();
        ExecuteInlineWorker worker = new ExecuteInlineWorker(p -> new WorkerCapacity.Snapshot(100L, 50L, 1000L), () -> {
            probes.incrementAndGet();
            probeThread.set(Thread.currentThread().getName());
            try { poll.await(5, TimeUnit.SECONDS); }
            catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            throw new IllegalStateException("synthetic unsupported runtime");
        });
        Thread running = new Thread(() -> {
            try { worker.execute(Arrays.asList("worker", "http://127.0.0.1:" + server.getAddress().getPort(),
                    "1", "--cache", root.toString(), "--max-shards", "3", "--capacity-session-limit", "16")); }
            catch (Throwable e) { error.set(e); }
        });
        try {
            running.start();
            assertTrue(poll.await(5, TimeUnit.SECONDS));
            assertTrue("capacity sender must not await long polling", heartbeat.await(5, TimeUnit.SECONDS));
            assertEquals("uuid=" + payload.get().get("hostId") + "&affinityLimit=3", query.get());
            assertEquals(100, ((Number) payload.get().get("ramTotalBytes")).intValue());
            assertEquals(0, ((Number) payload.get().get("activeSessions")).intValue());
            assertEquals(8, ((Number) payload.get().get("sessionLimit")).intValue());
            assertEquals(16, ((Number) payload.get().get("capacitySessionLimit")).intValue());
            assertEquals(3, ((Number) payload.get().get("affinityLimit")).intValue());
            assertEquals(false, payload.get().get("nativeReady"));
            assertEquals(false, payload.get().get("supportsRuntime"));
            assertNull(payload.get().get("gpuAllocatedBytes"));
            assertEquals(1, probes.get());
            assertEquals("device-capacity-ping", probeThread.get());
        } finally {
            worker.stop();
            release.countDown();
            running.join(5000);
            server.stop(0);
            http.shutdownNow();
            Files.deleteIfExists(root.resolve("tasks"));
            Files.deleteIfExists(root);
        }
        assertFalse("worker and reporter must stop", running.isAlive());
        assertNull(error.get());
    }

    @Test public void unreservedDagTransferDisablesAdmissionAndFailureReleasesIt() throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        ExecutorService http = Executors.newCachedThreadPool();
        server.setExecutor(http);
        CountDownLatch transfer = new CountDownLatch(1), heartbeat = new CountDownLatch(1),
                release = new CountDownLatch(1), returnedToPolling = new CountDownLatch(1);
        java.util.concurrent.atomic.AtomicInteger polls = new java.util.concurrent.atomic.AtomicInteger();
        AtomicReference<Map<?, ?>> payload = new AtomicReference<>();
        AtomicReference<Throwable> error = new AtomicReference<>();
        server.createContext("/pings/open", exchange -> {
            if (polls.getAndIncrement() == 0) {
                byte[] body = ("{\"status\":\"SUCCESS\",\"data\":{\"uuid\":\"dag-test\","
                        + "\"ruleEngineInput\":{\"arrays\":[{\"binaryFile\":\"/models/dag.bin\"}]}}}")
                        .getBytes(java.nio.charset.StandardCharsets.UTF_8);
                exchange.sendResponseHeaders(200, body.length);
                try (java.io.OutputStream out = exchange.getResponseBody()) { out.write(body); }
            } else {
                returnedToPolling.countDown();
                exchange.sendResponseHeaders(204, -1);
                exchange.close();
            }
        });
        server.createContext("/binary/stat", exchange -> {
            byte[] body = "{\"size\":4,\"mtime\":1}".getBytes(java.nio.charset.StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, body.length);
            try (java.io.OutputStream out = exchange.getResponseBody()) { out.write(body); }
        });
        server.createContext("/binary/fetch", exchange -> {
            transfer.countDown();
            try { release.await(10, TimeUnit.SECONDS); }
            catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            exchange.sendResponseHeaders(500, -1);
            exchange.close();
        });
        server.createContext("/task/complete", exchange -> {
            byte[] body = "{}".getBytes(java.nio.charset.StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, body.length);
            try (java.io.OutputStream out = exchange.getResponseBody()) { out.write(body); }
        });
        server.createContext("/pings/capacity", exchange -> {
            payload.set(new ObjectMapper().readValue(exchange.getRequestBody(), Map.class));
            exchange.sendResponseHeaders(204, -1);
            exchange.close();
            heartbeat.countDown();
        });
        server.start();
        Path root = Paths.get("target", "capacity-dag-" + UUID.randomUUID()).toAbsolutePath();
        ExecuteInlineWorker worker = new ExecuteInlineWorker(p -> {
            try { transfer.await(5, TimeUnit.SECONDS); }
            catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            return new WorkerCapacity.Snapshot(100L, 50L, 1000L);
        }, () -> {});
        Thread running = new Thread(() -> {
            try { worker.execute(Arrays.asList("worker", "http://127.0.0.1:" + server.getAddress().getPort(),
                    "1", "--cache", root.toString())); }
            catch (Throwable e) { error.set(e); }
        });
        try {
            running.start();
            assertTrue(transfer.await(5, TimeUnit.SECONDS));
            assertTrue(heartbeat.await(5, TimeUnit.SECONDS));
            assertEquals(false, payload.get().get("acceptingNewWork"));
            assertEquals(50, ((Number) payload.get().get("ramAvailableBytes")).intValue());
            assertNull(payload.get().get("ramAllocatedBytes"));
            assertTrue(!worker.acceptingNewWork());
            release.countDown();
            assertTrue(returnedToPolling.await(5, TimeUnit.SECONDS));
            assertTrue("failed unplanned work must release admission gate", worker.acceptingNewWork());
        } finally {
            worker.stop();
            release.countDown();
            running.join(5000);
            server.stop(0);
            http.shutdownNow();
            if (Files.exists(root)) {
                try (java.util.stream.Stream<Path> paths = Files.walk(root)) {
                    for (Path p : (Iterable<Path>) paths.sorted(java.util.Comparator.reverseOrder())::iterator)
                        Files.deleteIfExists(p);
                }
            }
        }
        assertTrue(!running.isAlive());
        assertNull(error.get());
    }
}

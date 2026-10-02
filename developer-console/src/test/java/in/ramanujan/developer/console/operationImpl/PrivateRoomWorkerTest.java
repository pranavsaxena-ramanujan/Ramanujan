package in.ramanujan.developer.console.operationImpl;

import com.sun.net.httpserver.HttpServer;
import org.junit.Test;
import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import static org.junit.Assert.*;

public class PrivateRoomWorkerTest {
    @Test public void immediateEmptyPollIsPacedAndStopInterruptsIdleWait() throws Exception {
        HttpServer gateway = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        AtomicInteger polls = new AtomicInteger();
        CountDownLatch firstPoll = new CountDownLatch(1);
        gateway.createContext("/worker/test_bearer/pings/open", exchange -> {
            polls.incrementAndGet(); firstPoll.countDown();
            byte[] response = "{\"status\":\"SUCCESS\",\"data\":null}".getBytes("UTF-8");
            exchange.sendResponseHeaders(200, response.length);
            try (OutputStream out = exchange.getResponseBody()) { out.write(response); }
        });
        gateway.start();
        Path cache = Paths.get("target", "private-room-test", UUID.randomUUID().toString()).toAbsolutePath();
        Files.createDirectories(cache);
        ExecuteInlineWorker worker = new ExecuteInlineWorker();
        AtomicReference<Throwable> failure = new AtomicReference<>();
        Thread thread = new Thread(() -> {
            try {
                worker.execute(Arrays.asList("worker", "http://127.0.0.1:" + gateway.getAddress().getPort()
                        + "/worker/test_bearer", "1", "--cache", cache.toString()));
            } catch (Throwable e) { failure.set(e); }
        });
        try {
            thread.start(); assertTrue(firstPoll.await(5, TimeUnit.SECONDS));
            Thread.sleep(350);
            worker.stop(); thread.join(1000);
            assertFalse("stop must interrupt polling/backoff", thread.isAlive());
            assertNull(failure.get());
            assertTrue("worker must continue polling", polls.get() >= 2);
            assertTrue("immediate empty responses must not busy-loop: " + polls.get(), polls.get() <= 5);
        } finally {
            worker.stop(); thread.interrupt(); thread.join(5000); gateway.stop(0);
            try (java.util.stream.Stream<Path> walk = Files.walk(cache)) {
                walk.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
        }
    }

    @Test public void alreadyLongPolledEmptyResponseGetsNoAdditionalDelay() throws Exception {
        long before = System.nanoTime();
        ExecuteInlineWorker.idleBackoff(before - TimeUnit.MILLISECONDS.toNanos(900));
        assertTrue("a completed long-poll must not sleep again",
                TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - before) < 80);
    }

    @Test public void binaryUploadPreservesTaskIdentityForPrivateGatewayAndLegacyServer() throws Exception {
        HttpServer gateway = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        List<String> requests = Collections.synchronizedList(new ArrayList<>());
        com.sun.net.httpserver.HttpHandler handler = exchange -> {
            requests.add(exchange.getRequestURI().toString());
            assertEquals(3, exchange.getRequestBody().read(new byte[3]));
            exchange.sendResponseHeaders(200, -1); exchange.close();
        };
        gateway.createContext("/worker/test_bearer/orchestrator/uploadBinary", handler);
        gateway.createContext("/orchestrator/uploadBinary", handler);
        gateway.start();
        Path root = Paths.get("target", "private-room-test", UUID.randomUUID().toString());
        Files.createDirectories(root);
        Path file = root.resolve("array.bin"); Files.write(file, new byte[]{1, 2, 3});
        ExecuteInlineWorker worker = new ExecuteInlineWorker();
        try {
            java.lang.reflect.Method upload = ExecuteInlineWorker.class.getDeclaredMethod(
                    "uploadBinaryFile", String.class, String.class, String.class, String.class);
            upload.setAccessible(true);
            String server = "http://127.0.0.1:" + gateway.getAddress().getPort();
            upload.invoke(worker, server + "/worker/test_bearer", "task-1", "array-1", file.toString());
            upload.invoke(worker, server, "task-1", "array-1", file.toString());
            assertEquals("/worker/test_bearer/orchestrator/uploadBinary?uuid=task-1&arrayId=array-1", requests.get(0));
            assertEquals("/orchestrator/uploadBinary?uuid=task-1&arrayId=array-1", requests.get(1));
        } finally {
            worker.stop(); gateway.stop(0); Files.deleteIfExists(file); Files.deleteIfExists(root);
        }
    }

    @Test public void scopedPollingRedactsCredentialAndStopTerminates() throws Exception {
        HttpServer gateway = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        CountDownLatch polled = new CountDownLatch(1);
        AtomicReference<String> path = new AtomicReference<>();
        gateway.createContext("/worker/test_bearer/pings/open", exchange -> {
            path.set(exchange.getRequestURI().toString()); polled.countDown();
            byte[] response = "{\"status\":\"SUCCESS\",\"data\":null}".getBytes("UTF-8");
            exchange.sendResponseHeaders(200, response.length);
            try (OutputStream out = exchange.getResponseBody()) { out.write(response); }
        });
        gateway.start();
        Path cache = Paths.get("target", "private-room-test", UUID.randomUUID().toString()).toAbsolutePath();
        Files.createDirectories(cache);
        PrintStream originalOut = System.out, originalErr = System.err;
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        ExecuteInlineWorker worker = new ExecuteInlineWorker();
        Thread thread = new Thread(() -> {
            try {
                worker.execute(Arrays.asList("worker", "http://127.0.0.1:" + gateway.getAddress().getPort()
                        + "/worker/test_bearer", "1", "--cache", cache.toString(), "--max-shards", "2", "--llm-sessions", "8"));
            } catch (IOException e) { throw new UncheckedIOException(e); }
        });
        try {
            System.setOut(new PrintStream(captured)); System.setErr(new PrintStream(captured));
            thread.start();
            assertTrue(polled.await(5, TimeUnit.SECONDS));
            worker.stop(); thread.join(5000);
            assertFalse(thread.isAlive());
            assertTrue(path.get().startsWith("/worker/test_bearer/pings/open?uuid="));
            assertTrue(path.get().contains("affinityLimit=2"));
            assertFalse(captured.toString("UTF-8").contains("test_bearer"));
        } finally {
            worker.stop(); thread.interrupt(); thread.join(5000);
            System.setOut(originalOut); System.setErr(originalErr);
            gateway.stop(0);
            try (java.util.stream.Stream<Path> walk = Files.walk(cache)) {
                walk.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
        }
    }
}

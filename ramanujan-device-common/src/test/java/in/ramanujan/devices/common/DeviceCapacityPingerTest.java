package in.ramanujan.devices.common;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import org.junit.Test;

import java.net.InetSocketAddress;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.*;

public class DeviceCapacityPingerTest {
    @Test
    public void preflightPrecedesFirstSharedCapacityPingAndCloseStopsSender() throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        ExecutorService http = Executors.newCachedThreadPool();
        server.setExecutor(http);
        CountDownLatch received = new CountDownLatch(1);
        AtomicBoolean preflight = new AtomicBoolean();
        AtomicReference<Map<?, ?>> request = new AtomicReference<>();
        AtomicReference<Throwable> error = new AtomicReference<>();
        server.createContext("/gateway/pings/capacity", exchange -> {
            request.set(new ObjectMapper().readValue(exchange.getRequestBody(), Map.class));
            exchange.sendResponseHeaders(204, -1);
            exchange.close();
            received.countDown();
        });
        server.start();
        DeviceCapacityPinger pinger = new DeviceCapacityPinger(
                "http://127.0.0.1:" + server.getAddress().getPort() + "/gateway",
                () -> {
                    assertTrue("native preflight must complete before the first report", preflight.get());
                    return Collections.<String, Object>singletonMap("hostId", "device-1");
                },
                () -> preflight.set(true), error::set);
        try {
            pinger.start();
            assertTrue("first capacity ping should be immediate", received.await(5, TimeUnit.SECONDS));
            assertEquals("device-1", request.get().get("hostId"));
            assertNull(error.get());
        } finally {
            pinger.close();
            server.stop(0);
            http.shutdownNow();
        }
    }
}

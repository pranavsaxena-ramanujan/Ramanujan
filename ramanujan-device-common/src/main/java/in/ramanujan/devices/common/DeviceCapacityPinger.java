package in.ramanujan.devices.common;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.Supplier;

/** Owns the shared capacity-ping cadence, preflight ordering, and HTTP transport. */
public final class DeviceCapacityPinger implements AutoCloseable {
    private static final long PERIOD_SECONDS = 10;
    private static final int CONNECT_TIMEOUT_MILLIS = 3000;
    private static final int READ_TIMEOUT_MILLIS = 3000;
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final String url;
    private final Supplier<Map<String, Object>> snapshot;
    private final Runnable nativePreflight;
    private final Consumer<Throwable> errors;
    private final Set<HttpURLConnection> connections = ConcurrentHashMap.newKeySet();
    private final ScheduledExecutorService executor;
    private volatile boolean stopped;
    private volatile boolean nativeProbeFailed;

    public DeviceCapacityPinger(String serverUrl, Supplier<Map<String, Object>> snapshot,
                                Runnable nativePreflight, Consumer<Throwable> errors) {
        this.url = DeviceProtocol.capacityUrl(Objects.requireNonNull(serverUrl, "serverUrl"));
        this.snapshot = Objects.requireNonNull(snapshot, "snapshot");
        this.nativePreflight = Objects.requireNonNull(nativePreflight, "nativePreflight");
        this.errors = Objects.requireNonNull(errors, "errors");
        this.executor = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread thread = new Thread(r, "device-capacity-ping");
            thread.setDaemon(true);
            return thread;
        });
    }

    public void start() {
        executor.execute(() -> {
            if (stopped) return;
            try {
                nativePreflight.run();
            } catch (Exception | LinkageError error) {
                nativeProbeFailed = true;
                reportError(error);
            }
            if (!stopped) executor.scheduleWithFixedDelay(this::sendSafely, 0, PERIOD_SECONDS, TimeUnit.SECONDS);
        });
    }

    private void sendSafely() {
        if (stopped) return;
        try {
            Map<String, Object> value = snapshot.get();
            if (nativeProbeFailed && value.get("nativeReady") == null) {
                value.put("nativeReady", false);
                value.put("supportsRuntime", false);
            }
            post(MAPPER.writeValueAsBytes(value));
        } catch (Exception | LinkageError error) {
            reportError(error);
        }
    }

    private void post(byte[] body) throws IOException {
        HttpURLConnection connection = (HttpURLConnection) new URL(url).openConnection();
        connections.add(connection);
        try {
            if (stopped) throw new IOException("capacity pinger stopped");
            connection.setRequestMethod("POST");
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setDoOutput(true);
            connection.setConnectTimeout(CONNECT_TIMEOUT_MILLIS);
            connection.setReadTimeout(READ_TIMEOUT_MILLIS);
            connection.setFixedLengthStreamingMode(body.length);
            try (OutputStream output = connection.getOutputStream()) { output.write(body); }
            int status = connection.getResponseCode();
            if (status < 200 || status >= 300) throw new IOException("capacity request failed with HTTP " + status);
        } finally {
            connections.remove(connection);
            connection.disconnect();
        }
    }

    private void reportError(Throwable error) {
        if (!stopped) errors.accept(error);
    }

    @Override
    public void close() {
        stopped = true;
        executor.shutdownNow();
        for (HttpURLConnection connection : connections) connection.disconnect();
    }
}

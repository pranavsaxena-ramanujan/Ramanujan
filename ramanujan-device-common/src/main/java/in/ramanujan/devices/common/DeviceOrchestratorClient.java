package in.ramanujan.devices.common;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The device side of every orchestrator exchange: task polling, completion, result upload and
 * weight transfer. Cluster/room routing is server-side; a private room is just a scoped server URL.
 */
public final class DeviceOrchestratorClient implements AutoCloseable {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final String serverUrl;
    private final Set<HttpURLConnection> connections = ConcurrentHashMap.newKeySet();
    private volatile boolean closed;

    public DeviceOrchestratorClient(String serverUrl) {
        this.serverUrl = DeviceProtocol.trimSlash(serverUrl);
    }

    public String serverUrl() {
        return serverUrl;
    }

    /** Long-polls for the next task; returns the orchestrator response, or null for an empty body. */
    public Map<String, Object> pollTask(String deviceId, Integer affinityLimit) throws IOException {
        return postJson(DeviceProtocol.openUrl(serverUrl, deviceId, affinityLimit), new byte[0]);
    }

    public void completeTask(String taskId, String deviceId, Map<String, Object> data, String error)
            throws IOException {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("uuid", taskId);
        payload.put("hostId", deviceId);
        payload.put("data", data);
        if (error != null) payload.put("error", error);
        postJson(DeviceProtocol.taskCompleteUrl(serverUrl), MAPPER.writeValueAsBytes(payload));
    }

    /** Uploads a RETURN()-marked array as raw little-endian bytes. */
    public void uploadBinary(String taskId, String arrayId, byte[] data) throws IOException {
        HttpURLConnection conn = open(DeviceProtocol.uploadBinaryUrl(serverUrl, taskId, arrayId));
        try {
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/octet-stream");
            conn.setDoOutput(true);
            conn.setConnectTimeout(5000);
            conn.setReadTimeout(120_000);
            conn.setFixedLengthStreamingMode(data.length);
            try (OutputStream os = conn.getOutputStream()) {
                os.write(data);
            }
            int code = conn.getResponseCode();
            InputStream is = code < 400 ? conn.getInputStream() : conn.getErrorStream();
            if (is != null) is.close();
            if (code >= 400) throw new IOException("uploadBinary for arrayId=" + arrayId + " failed with HTTP " + code);
        } finally {
            release(conn);
        }
    }

    /** Size and mtime of an orchestrator-served immutable file. */
    @SuppressWarnings("unchecked")
    public Map<String, Object> binaryStat(String serverPath) throws IOException {
        HttpURLConnection conn = open(DeviceProtocol.binaryStatUrl(serverUrl, serverPath));
        conn.setConnectTimeout(10_000);
        conn.setReadTimeout(60_000);
        try {
            int code = conn.getResponseCode();
            if (code != 200) throw new IOException("binary stat for " + serverPath + " failed with HTTP " + code);
            try (InputStream in = conn.getInputStream()) {
                return MAPPER.readValue(in, Map.class);
            }
        } finally {
            release(conn);
        }
    }

    /** Opens a streaming download; the caller reads it and must {@link #release} the connection. */
    public HttpURLConnection openBinaryFetch(String serverPath) throws IOException {
        HttpURLConnection conn = open(DeviceProtocol.binaryFetchUrl(serverUrl, serverPath));
        conn.setConnectTimeout(10_000);
        conn.setReadTimeout(600_000);
        return conn;
    }

    public void release(HttpURLConnection conn) {
        if (conn == null) return;
        connections.remove(conn);
        conn.disconnect();
    }

    /** Aborts in-flight requests and rejects new ones. */
    @Override
    public void close() {
        closed = true;
        for (HttpURLConnection connection : connections) connection.disconnect();
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> postJson(String url, byte[] body) throws IOException {
        HttpURLConnection conn = open(url);
        try {
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setDoOutput(true);
            conn.setConnectTimeout(5000);
            conn.setReadTimeout(120_000);
            conn.setFixedLengthStreamingMode(body.length);
            try (OutputStream os = conn.getOutputStream()) {
                os.write(body);
            }
            int code = conn.getResponseCode();
            if (code < 200 || code >= 300) throw new IOException("worker request failed with HTTP " + code);
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            try (InputStream is = conn.getInputStream()) {
                if (is == null) return null;
                byte[] buffer = new byte[4096];
                int n;
                while ((n = is.read(buffer)) != -1) bytes.write(buffer, 0, n);
            }
            if (bytes.size() == 0) return null;
            return MAPPER.readValue(new String(bytes.toByteArray(), StandardCharsets.UTF_8), Map.class);
        } finally {
            release(conn);
        }
    }

    private HttpURLConnection open(String url) throws IOException {
        if (closed) throw new IOException("device client stopped");
        HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
        conn.setInstanceFollowRedirects(false);
        connections.add(conn);
        if (closed) { release(conn); throw new IOException("device client stopped"); }
        return conn;
    }
}

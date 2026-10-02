package in.ramanujan.cluster.client.common;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.util.LinkedHashMap;
import java.util.Map;

public final class JoinClient {
    private static final ObjectMapper JSON = new ObjectMapper();

    public static String portal(String value) {
        try {
            URI uri = URI.create(value.trim().replaceAll("/+$", ""));
            if (!("https".equals(uri.getScheme()) || "http".equals(uri.getScheme()))
                    || uri.getHost() == null || uri.getUserInfo() != null
                    || uri.getQuery() != null || uri.getFragment() != null) {
                throw new IllegalArgumentException();
            }
            return uri.toASCIIString();
        } catch (RuntimeException e) {
            throw new IllegalArgumentException("Enter an absolute HTTP(S) portal URL without credentials, query or fragment.");
        }
    }

    public Registration join(String portal, String room, String secret, String name, String platform) throws IOException {
        return join(portal, room, secret, name, platform, false);
    }

    public Registration join(String portal, String room, String secret, String name, String platform,
                             boolean allowLocalHttp) throws IOException {
        portal = portal(portal);
        validateTransport(portal, allowLocalHttp);
        if (room.trim().isEmpty() || secret.isEmpty() || name.trim().isEmpty()) {
            throw new IllegalArgumentException("Room ID, join secret and device name are required.");
        }

        Map<String, String> body = new LinkedHashMap<>();
        body.put("roomId", room.trim()); body.put("joinSecret", secret);
        body.put("name", name.trim()); body.put("platform", platform);
        HttpURLConnection connection = (HttpURLConnection) URI.create(portal + "/api/devices/join").toURL().openConnection();
        connection.setInstanceFollowRedirects(false);
        connection.setConnectTimeout(10000); connection.setReadTimeout(30000);
        connection.setRequestMethod("POST"); connection.setDoOutput(true);
        connection.setRequestProperty("Content-Type", "application/json");
        byte[] bytes = JSON.writeValueAsBytes(body);
        connection.setFixedLengthStreamingMode(bytes.length);
        try {
            try (OutputStream out = connection.getOutputStream()) { out.write(bytes); }
            int status = connection.getResponseCode();
            if (status != 200 && status != 201) throw new IOException("Join refused (HTTP " + status + "). Check the room and join secret.");
            Map<?, ?> response;
            try (InputStream in = connection.getInputStream()) { response = JSON.readValue(in, Map.class); }
            String device = field(response, "deviceId"), cluster = field(response, "clusterId");
            String worker = field(response, "workerUrl");
            validateWorkerUrl(portal, worker);
            return new Registration(device, cluster, worker.replaceAll("/+$", ""));
        } catch (IOException e) {
            // Network exceptions can include a complete URL; never relay server-controlled text.
            if (e.getMessage() != null && e.getMessage().startsWith("Join refused")) throw e;
            throw new IOException("Unable to join: verify portal connectivity and its device-join response.");
        } finally { connection.disconnect(); }
    }

    public static void validateTransport(String portal, boolean allowLocalHttp) {
        URI uri = URI.create(portal(portal));
        if ("https".equals(uri.getScheme())) return;
        String host = uri.getHost().toLowerCase(java.util.Locale.ROOT);
        if (allowLocalHttp && isLocalHost(host)) return;
        throw new IllegalArgumentException("Use HTTPS. Development HTTP requires explicit opt-in and localhost or a private IP address.");
    }

    private static boolean isLocalHost(String host) {
        if ("localhost".equals(host) || host.endsWith(".localhost")) return true;
        if (host.startsWith("[") && host.endsWith("]")) {
            String literal = host.substring(1, host.length() - 1);
            try {
                java.net.InetAddress address = java.net.InetAddress.getByName(literal);
                byte[] bytes = address.getAddress();
                return address.isLoopbackAddress() || address.isLinkLocalAddress()
                        || (bytes.length == 16 && (bytes[0] & 0xfe) == 0xfc);
            } catch (IOException e) { return false; }
        }
        if (!host.matches("[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+")) return false;
        String[] parts = host.split("\\.");
        int[] octets = new int[4];
        try {
            for (int i = 0; i < 4; i++) {
                octets[i] = Integer.parseInt(parts[i]);
                if (octets[i] > 255 || parts[i].length() > 3) return false;
            }
        } catch (NumberFormatException e) { return false; }
        return octets[0] == 127 || octets[0] == 10
                || (octets[0] == 172 && octets[1] >= 16 && octets[1] <= 31)
                || (octets[0] == 192 && octets[1] == 168)
                || (octets[0] == 169 && octets[1] == 254);
    }

    private static String field(Map<?, ?> response, String key) throws IOException {
        Object value = response.get(key);
        if (!(value instanceof String) || ((String) value).trim().isEmpty()) throw new IOException("Invalid device registration.");
        return (String) value;
    }

    public static void validateWorkerUrl(String portal, String worker) throws IOException {
        try {
            URI origin = URI.create(portal(portal)), uri = URI.create(worker);
            String prefix = origin.getPath().replaceAll("/+$", "") + "/worker/";
            String path = uri.getPath();
            if (!origin.getScheme().equals(uri.getScheme()) || !origin.getHost().equalsIgnoreCase(uri.getHost())
                    || origin.getPort() != uri.getPort() || uri.getUserInfo() != null
                    || uri.getQuery() != null || uri.getFragment() != null || path == null
                    || !path.startsWith(prefix) || !path.substring(prefix.length()).matches("[A-Za-z0-9_-]+")) {
                throw new IllegalArgumentException();
            }
        } catch (RuntimeException e) { throw new IOException("Portal returned an invalid scoped worker URL."); }
    }

    public static String redact(String line) {
        return line.replaceAll("(?i)(https?://[^\\s]+)?/worker/[^\\s/?\"']+", "[private room gateway]");
    }
}

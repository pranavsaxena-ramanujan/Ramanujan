package in.ramanujan.developer.console.operationImpl;

import com.fasterxml.jackson.databind.ObjectMapper;
import in.ramanujan.pojo.RuleEngineInput;
import in.ramanujan.pojo.ruleEngineInputUnitsExt.array.Array;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Collection;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Worker-side copies of the binary arrays a homelab task references.
 *
 * Immutable binaries (model weights) are cached persistently under {@code root/objects}
 * and reused while the server reports the same size and mtime, so a worker downloads
 * each of its shards once. Mutable runtime state (names the native runtime treats as
 * writable: hidden.bin, *_state.bin, *_k_cache.bin, *_v_cache.bin) is fetched fresh into a
 * per-task directory and keeps its file name so native keeps treating it as mutable.
 */
final class WorkerBinaryCache implements AutoCloseable {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final Path root;
    private final String serverUrl;
    private final Map<String, Object> locks = new ConcurrentHashMap<>();
    // Server paths already validated by this process; model weights are immutable while serving.
    private final Map<String, Path> validated = new ConcurrentHashMap<>();
    private final java.util.Set<HttpURLConnection> connections = ConcurrentHashMap.newKeySet();
    private volatile boolean closed;
    private final long diskReserveBytes;
    private long reservedDownloads;
    private final WorkerShardManifest shards;

    @Override
    public void close() {
        closed = true;
        for (HttpURLConnection connection : connections) connection.disconnect();
    }

    private HttpURLConnection connection(String route) throws IOException {
        if (closed) throw new IOException("binary cache stopped");
        HttpURLConnection conn = (HttpURLConnection) new URL(serverUrl + route).openConnection();
        conn.setInstanceFollowRedirects(false);
        connections.add(conn);
        if (closed) { conn.disconnect(); throw new IOException("binary cache stopped"); }
        return conn;
    }

    WorkerBinaryCache(Path root, String serverUrl) {
        this(root, serverUrl, 64L << 20);
    }

    WorkerBinaryCache(Path root, String serverUrl, long diskReserveBytes) {
        if (diskReserveBytes < 0) throw new IllegalArgumentException("disk reserve must be >= 0");
        this.root = root.toAbsolutePath();
        this.serverUrl = serverUrl;
        this.diskReserveBytes = diskReserveBytes;
        this.shards = new WorkerShardManifest(this.root, serverUrl);
    }

    WorkerShardManifest shards() { return shards; }

    static void checkDiskSpace(long available, long downloadBytes, long reserveBytes, long reserved) throws IOException {
        if (downloadBytes < 0 || available < reserveBytes || reserved > available - reserveBytes
                || downloadBytes > available - reserveBytes - reserved) {
            throw new IOException("Insufficient cache disk space: download requires " + downloadBytes
                    + " bytes, available=" + available + ", reserve=" + reserveBytes
                    + ", concurrent reservations=" + reserved + "; free disk space or reduce --disk-reserve-bytes");
        }
    }

    static boolean isMutable(String path) {
        String name = Paths.get(path).getFileName().toString();
        return name.equals("hidden.bin") || name.endsWith("_state.bin")
                || name.endsWith("_k_cache.bin") || name.endsWith("_v_cache.bin");
    }

    /** Rewrites every binaryFile in {@code rei} to a local path; mutable files go under {@code taskDir}. */
    void localize(RuleEngineInput rei, Path taskDir) throws IOException {
        if (rei.getArrays() == null) return;
        int index = 0;
        Map<String, Path> fetched = new LinkedHashMap<>();
        for (Array array : rei.getArrays()) {
            String serverPath = array.getBinaryFile();
            if (serverPath == null || serverPath.isEmpty()) continue;
            Path local = fetched.get(serverPath);
            if (local == null) {
                if (isMutable(serverPath)) {
                    local = taskDir.resolve(String.valueOf(index++)).resolve(Paths.get(serverPath).getFileName());
                    Files.createDirectories(local.getParent());
                    download(serverPath, local, -1);
                } else {
                    local = cached(serverPath);
                }
                fetched.put(serverPath, local);
            }
            array.setBinaryFile(local.toString());
        }
    }

    Path cached(String serverPath) throws IOException {
        Path known = validated.get(serverPath);
        if (known != null) return known;
        // Equal server paths/stat metadata in different rooms must never share model bytes.
        String key = sha256(serverUrl + "\n" + serverPath);
        Object lock = locks.computeIfAbsent(key, k -> new Object());
        synchronized (lock) {
            Path file = cachedAfterStat(serverPath, key);
            validated.put(serverPath, file);
            return file;
        }
    }

    /** Checks all assigned immutable weights before fetching; already validated/cached files cost zero. */
    void preflight(Collection<String> serverPaths) throws IOException {
        long required = 0;
        for (String serverPath : new java.util.LinkedHashSet<>(serverPaths)) {
            if (validated.containsKey(serverPath)) continue;
            String key = sha256(serverUrl + "\n" + serverPath);
            Path directory = root.resolve("objects").resolve(key);
            Map<String, Object> stat = getJson("/binary/stat?path=" + encode(serverPath));
            long size = ((Number) stat.get("size")).longValue();
            long mtime = ((Number) stat.get("mtime")).longValue();
            if (size < 0) throw new IOException("invalid binary size for " + serverPath + ": " + size);
            Path file = directory.resolve(Paths.get(serverPath).getFileName());
            if (matches(file, directory.resolve("meta.json"), serverPath, size, mtime)) continue;
            try { required = Math.addExact(required, size); }
            catch (ArithmeticException e) { throw new IOException("assigned immutable weight bytes overflow disk budget", e); }
        }
        if (required == 0) return;
        Files.createDirectories(root);
        synchronized (this) {
            checkDiskSpace(Files.getFileStore(root).getUsableSpace(), required, diskReserveBytes, reservedDownloads);
        }
    }

    private Path cachedAfterStat(String serverPath, String key) throws IOException {
        Path directory = root.resolve("objects").resolve(key);
        Path file = directory.resolve(Paths.get(serverPath).getFileName());
        Path meta = directory.resolve("meta.json");
        Map<String, Object> stat = getJson("/binary/stat?path=" + encode(serverPath));
        long size = ((Number) stat.get("size")).longValue();
        long mtime = ((Number) stat.get("mtime")).longValue();
        if (matches(file, meta, serverPath, size, mtime)) return file;
        Files.createDirectories(directory);
        Files.deleteIfExists(meta);
        System.err.println("[Worker] caching " + serverPath + " (" + (size >> 20) + " MiB)");
        download(serverPath, file, size);
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("path", serverPath);
        value.put("size", size);
        value.put("mtime", mtime);
        Path temporary = directory.resolve("meta.json.partial");
        Files.write(temporary, MAPPER.writeValueAsBytes(value));
        Files.move(temporary, meta, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        return file;
    }

    private static boolean matches(Path file, Path meta, String serverPath, long size, long mtime) throws IOException {
        if (!Files.isRegularFile(file) || Files.size(file) != size || !Files.isRegularFile(meta)) return false;
        Map<?, ?> cachedMeta = MAPPER.readValue(meta.toFile(), Map.class);
        return serverPath.equals(cachedMeta.get("path"))
                && cachedMeta.get("size") instanceof Number && ((Number) cachedMeta.get("size")).longValue() == size
                && cachedMeta.get("mtime") instanceof Number && ((Number) cachedMeta.get("mtime")).longValue() == mtime;
    }

    /** Streams one file; a truncated or oversized transfer is never moved into place. */
    private void download(String serverPath, Path destination, long expectedSize) throws IOException {
        Path temporary = destination.resolveSibling(destination.getFileName() + ".partial");
        HttpURLConnection conn = null;
        long reservation = 0;
        try {
            if (expectedSize >= 0) {
                reserve(destination, expectedSize);
                reservation = expectedSize;
            }
            conn = connection("/binary/fetch?path=" + encode(serverPath));
            conn.setConnectTimeout(10_000);
            conn.setReadTimeout(600_000);
            int code = conn.getResponseCode();
            if (code != 200) {
                throw new IOException("fetch " + serverPath + " failed with HTTP " + code);
            }
            long length = conn.getContentLengthLong();
            if (length < 0 || (expectedSize >= 0 && length != expectedSize)) {
                throw new IOException("fetch " + serverPath + " length " + length + " != expected " + expectedSize);
            }
            if (expectedSize < 0) {
                reserve(destination, length);
                reservation = length;
            }
            long copied = 0;
            try (InputStream in = conn.getInputStream(); OutputStream out = Files.newOutputStream(temporary)) {
                byte[] buffer = new byte[1 << 20];
                int n;
                while ((n = in.read(buffer)) != -1) {
                    copied += n;
                    if (copied > length) break;
                    out.write(buffer, 0, n);
                }
            }
            if (copied != length) {
                throw new IOException("fetch " + serverPath + " received " + copied + " of " + length + " bytes");
            }
            Files.move(temporary, destination, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } finally {
            synchronized (this) { reservedDownloads -= reservation; }
            try { Files.deleteIfExists(temporary); }
            finally {
                if (conn != null) { connections.remove(conn); conn.disconnect(); }
            }
        }
    }

    private synchronized void reserve(Path destination, long bytes) throws IOException {
        checkDiskSpace(Files.getFileStore(destination.getParent()).getUsableSpace(),
                bytes, diskReserveBytes, reservedDownloads);
        reservedDownloads += bytes;
    }

    static final class CacheInventory {
        final List<Map<String, Object>> files;
        final boolean truncated;

        CacheInventory(List<Map<String, Object>> files, boolean truncated) {
            this.files = files;
            this.truncated = truncated;
        }
    }

    CacheInventory inventory() { return inventory(4096); }

    CacheInventory inventory(int limit) {
        if (limit < 0 || limit > 4096) throw new IllegalArgumentException("cache inventory limit must be 0..4096");
        List<Map<String, Object>> files = new ArrayList<>();
        boolean incomplete = false;
        Path objects = root.resolve("objects");
        if (Files.notExists(objects)) return new CacheInventory(files, false);
        if (!Files.isDirectory(objects, java.nio.file.LinkOption.NOFOLLOW_LINKS)) return new CacheInventory(files, true);
        try (java.nio.file.DirectoryStream<Path> directories = Files.newDirectoryStream(objects)) {
            for (Path directory : directories) {
                if (!Files.isDirectory(directory, java.nio.file.LinkOption.NOFOLLOW_LINKS)) continue;
                Path meta = directory.resolve("meta.json");
                if (!Files.isRegularFile(meta, java.nio.file.LinkOption.NOFOLLOW_LINKS)) continue;
                try {
                    Map<?, ?> value = MAPPER.readValue(meta.toFile(), Map.class);
                    if (!(value.get("path") instanceof String) || !(value.get("size") instanceof Number)
                            || !(value.get("mtime") instanceof Number)) continue;
                    String path = (String) value.get("path");
                    // Only this gateway's objects count. Never expose bearer URLs or cross-room cache entries.
                    if (path.contains("://") || path.matches("(?i).*[/\\\\]worker[/\\\\][^/\\\\]+.*")
                            || !sha256(serverUrl + "\n" + path).equals(directory.getFileName().toString())) continue;
                    Path local = directory.resolve(Paths.get(path).getFileName());
                    if (!Files.isRegularFile(local, java.nio.file.LinkOption.NOFOLLOW_LINKS)) continue;
                    long bytes = Files.size(local);
                    if (bytes != ((Number) value.get("size")).longValue()) continue;
                    if (files.size() >= limit) return new CacheInventory(files, true);
                    Map<String, Object> fingerprint = new LinkedHashMap<>();
                    fingerprint.put("path", path);
                    fingerprint.put("bytes", bytes);
                    fingerprint.put("mtime", ((Number) value.get("mtime")).longValue());
                    files.add(fingerprint);
                } catch (IOException | RuntimeException e) {
                    incomplete = true;
                }
            }
        } catch (IOException | RuntimeException e) {
            incomplete = true;
        }
        return new CacheInventory(files, incomplete);
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> getJson(String route) throws IOException {
        HttpURLConnection conn = connection(route);
        conn.setConnectTimeout(10_000);
        conn.setReadTimeout(60_000);
        try {
            int code = conn.getResponseCode();
            if (code != 200) {
                throw new IOException(route + " failed with HTTP " + code);
            }
            try (InputStream in = conn.getInputStream()) {
                return MAPPER.readValue(in, Map.class);
            }
        } finally {
            connections.remove(conn);
            conn.disconnect();
        }
    }

    private static String encode(String value) throws IOException {
        return URLEncoder.encode(value, "UTF-8");
    }

    private static String sha256(String value) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder();
            for (byte b : digest) hex.append(String.format("%02x", b));
            return hex.toString();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }
}

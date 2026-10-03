package in.ramanujan.devices.common;

import com.fasterxml.jackson.databind.ObjectMapper;
import in.ramanujan.rule.engine.LlmSession;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Stream;

/** Shared device-capacity schema and platform memory sampling. */
public final class WorkerCapacity {
    public interface Provider {
        Snapshot sample(Path cacheRoot);
    }

    public static final class Snapshot {
        public final Long ramTotalBytes, ramAvailableBytes, diskAvailableBytes;

        public Snapshot(Long total, Long available, Long disk) {
            ramTotalBytes = nonnegative(total);
            ramAvailableBytes = nonnegative(available);
            diskAvailableBytes = nonnegative(disk);
        }
    }

    public static final class DesktopProvider implements Provider {
        public Snapshot sample(Path root) {
            Long total = null, available = null;
            try {
                Object bean = Class.forName("java.lang.management.ManagementFactory")
                        .getMethod("getOperatingSystemMXBean").invoke(null);
                Class<?> type = Class.forName("com.sun.management.OperatingSystemMXBean");
                total = (Long) type.getMethod("getTotalPhysicalMemorySize").invoke(bean);
                available = (Long) type.getMethod("getFreePhysicalMemorySize").invoke(bean);
            } catch (ReflectiveOperationException | RuntimeException ignored) {
                // Leave physical memory unknown on unsupported JVMs.
            }
            Long reclaimable = reclaimableMemory();
            if (reclaimable != null) available = total == null ? reclaimable : Math.min(total, reclaimable);
            Long disk = null;
            try { disk = Files.getFileStore(root).getUsableSpace(); }
            catch (Exception ignored) {}
            return new Snapshot(total, available, disk);
        }

        /** Linux MemAvailable, or macOS free + inactive + speculative pages; null when unknown. */
        static Long reclaimableMemory() {
            String os = System.getProperty("os.name", "").toLowerCase(java.util.Locale.ROOT);
            try {
                if (os.contains("linux")) {
                    for (String line : Files.readAllLines(java.nio.file.Paths.get("/proc/meminfo"))) {
                        if (line.startsWith("MemAvailable:")) return Long.parseLong(line.replaceAll("[^0-9]", "")) * 1024;
                    }
                } else if (os.contains("mac")) {
                    Process process = new ProcessBuilder("/usr/bin/vm_stat").redirectErrorStream(true).start();
                    StringBuilder text = new StringBuilder();
                    try (java.io.BufferedReader reader = new java.io.BufferedReader(
                            new java.io.InputStreamReader(process.getInputStream(), java.nio.charset.StandardCharsets.US_ASCII))) {
                        for (String line; (line = reader.readLine()) != null; ) text.append(line).append('\n');
                    }
                    if (!process.waitFor(3, java.util.concurrent.TimeUnit.SECONDS)) {
                        process.destroyForcibly();
                        return null;
                    }
                    return parseVmStat(text.toString());
                }
            } catch (Exception ignored) {
                // Fall back to the JVM's free-memory figure.
            }
            return null;
        }

        public static Long parseVmStat(String text) {
            java.util.regex.Matcher size = java.util.regex.Pattern.compile("page size of (\\d+) bytes").matcher(text);
            if (!size.find()) return null;
            long pages = 0;
            int found = 0;
            for (String name : new String[]{"Pages free", "Pages inactive", "Pages speculative"}) {
                java.util.regex.Matcher count = java.util.regex.Pattern.compile(name + ":\\s+(\\d+)\\.").matcher(text);
                if (count.find()) { pages += Long.parseLong(count.group(1)); found++; }
            }
            return found == 3 ? pages * Long.parseLong(size.group(1)) : null;
        }
    }

    private final Provider provider;
    private final Path root;
    private final String hostId;
    private final String bootId = UUID.randomUUID().toString();
    private long sequence;

    public WorkerCapacity(Provider provider, Path root, String hostId) {
        this.provider = java.util.Objects.requireNonNull(provider, "provider");
        this.root = java.util.Objects.requireNonNull(root, "root");
        this.hostId = java.util.Objects.requireNonNull(hostId, "hostId");
    }

    public Map<String, Object> sample(int sessions, boolean accepting) {
        return sample(sessions, accepting, java.util.Collections.emptyMap(), java.util.Collections.emptyMap());
    }

    public Map<String, Object> sample(int sessions, boolean accepting, Map<String, Integer> activePlans) {
        return sample(sessions, accepting, activePlans, java.util.Collections.emptyMap());
    }

    public Map<String, Object> sample(int sessions, boolean accepting, Map<String, Integer> activePlans,
                                      Map<String, Object> deviceDetails) {
        Snapshot snapshot;
        try { snapshot = provider.sample(root); }
        catch (RuntimeException e) { snapshot = new Snapshot(null, null, null); }
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("schemaVersion", 1);
        value.put("hostId", hostId);
        value.put("bootId", bootId);
        value.put("sequence", ++sequence);
        value.put("ramTotalBytes", snapshot.ramTotalBytes);
        value.put("ramAvailableBytes", snapshot.ramAvailableBytes);
        value.put("ramAllocatedBytes", null);
        value.put("diskAvailableBytes", snapshot.diskAvailableBytes);
        value.put("cacheBytes", cacheBytes(root));
        value.put("cachedFiles", java.util.Collections.emptyList());
        value.put("cachedFilesTruncated", false);
        value.put("gpuTotalBytes", null);
        value.put("gpuAvailableBytes", null);
        value.put("gpuResidentBytes", null);
        value.put("gpuAllocatedBytes", null);
        value.put("gpuMaxAllocationBytes", null);
        value.put("gpuDeviceName", null);
        value.put("gpuDeviceVendor", null);
        value.put("gpuDriverVersion", null);
        value.put("gpuRuntime", null);
        value.put("gpuRuntimeVersion", null);
        value.put("unifiedMemory", false);
        value.put("llmRuntimeAvailable", LlmSession.isLoaded());
        value.put("nativeReady", null);
        value.put("supportsRuntime", null);
        value.put("supportsStreaming", null);
        try {
            String json = LlmSession.capacityInfo();
            if (json != null) {
                Map<?, ?> gpu = new ObjectMapper().readValue(json, Map.class);
                for (String field : new String[]{"gpuTotalBytes", "gpuAvailableBytes", "gpuResidentBytes",
                        "gpuAllocatedBytes", "gpuMaxAllocationBytes", "gpuDeviceName", "gpuDeviceVendor",
                        "gpuDriverVersion", "gpuRuntime", "gpuRuntimeVersion", "unifiedMemory",
                        "nativeReady", "supportsRuntime", "supportsStreaming"})
                    if (gpu.containsKey(field)) value.put(field, gpu.get(field));
            }
        } catch (Exception | LinkageError ignored) {
            // Missing/older JNI libraries must not disable worker execution.
        }
        value.put("activeSessions", sessions);
        value.put("activePlans", new LinkedHashMap<>(activePlans));
        value.put("acceptingNewWork", accepting);
        for (Map.Entry<String, Object> detail : deviceDetails.entrySet()) {
            if ("cachedFiles".equals(detail.getKey()) || "cachedFilesTruncated".equals(detail.getKey())) {
                value.put(detail.getKey(), detail.getValue());
            } else if (value.containsKey(detail.getKey())) {
                throw new IllegalArgumentException("device capacity detail conflicts with schema field: " + detail.getKey());
            } else {
                value.put(detail.getKey(), detail.getValue());
            }
        }
        return value;
    }

    private static Long cacheBytes(Path root) {
        try (Stream<Path> files = Files.walk(root)) {
            long bytes = 0;
            for (java.util.Iterator<Path> it = files.iterator(); it.hasNext();) {
                Path path = it.next();
                if (Files.isRegularFile(path)) bytes = Math.addExact(bytes, Files.size(path));
            }
            return bytes;
        } catch (Exception e) { return null; }
    }

    private static Long nonnegative(Long value) { return value != null && value >= 0 ? value : null; }
}

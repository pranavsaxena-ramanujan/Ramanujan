package in.ramanujan.developer.console.operationImpl;

import com.fasterxml.jackson.databind.ObjectMapper;
import in.ramanujan.developer.console.Operation;
import in.ramanujan.devices.common.DeviceCapacityPinger;
import in.ramanujan.devices.common.DeviceOrchestratorClient;
import in.ramanujan.devices.common.WorkerCapacity;
import in.ramanujan.pojo.RuleEngineInput;
import in.ramanujan.pojo.ruleEngineInputUnitsExt.FunctionCall;
import in.ramanujan.rule.engine.NativeProcessor;
import in.ramanujan.rule.engine.LlmSession;
import in.ramanujan.rule.engine.RuleEngineInputProtoSerializer;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Local worker mode: polls a homelab server for tasks and executes them via
 * NativeProcessor on this machine.  Use this to triage whether a problem is
 * in the homelab server/orchestrator or in the Android GPU execution path.
 *
 * Usage:
 *   java -jar developer-console.jar worker [server-url] [num-threads] [options]
 *
 * Defaults:
 *   server-url   = RAMANUJAN_WORKER_URL, or http://localhost:8888
 *   num-threads  = number of available CPUs
 *
 * Options:
 *   --native-library NAME  native library loaded from $RAMANUJAN_WS (default "native";
 *                          use "native_llm" for the GGUF/Qwen runtime)
 *   --cache DIR            where binary arrays fetched from the server are cached
 *                          (default ~/.ramanujan/worker-cache)
 *   --max-shards N         most affinity groups (model shards) this worker accepts
 *   --shared-filesystem    read binary arrays at the server's paths instead of fetching
 *   --llm-sessions N       pinned native LLM stage sessions (default 8); new sessions at the
 *                          limit are rejected until an explicit /llm/close.
 *   --disk-reserve-bytes N free disk space preserved while downloading (default 64 MiB).
 *   --capacity-session-limit N opt-in total stage-session ceiling for trusted planId tasks
 *                          (1..1024, >= --llm-sessions; default same as --llm-sessions).
 *
 * Examples:
 *   java -jar developer-console.jar worker
 *   java -jar developer-console.jar worker http://localhost:8888 2
 *   java -jar developer-console.jar worker http://192.168.1.42:8888
 *
 * The worker continuously polls /pings/open, executes the ruleEngineInput
 * through NativeProcessor (same path as Android), and posts results to
 * /task/complete. Private-room launchers pass the scoped bearer URL via
 * RAMANUJAN_WORKER_URL rather than command-line arguments. Press Ctrl-C to stop.
 */
public class ExecuteInlineWorker implements Operation {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Object GPU_EXECUTION_LOCK = new Object();
    // Native keeps one process-wide weight cache; eviction must not overlap an execution.
    private static final ReentrantReadWriteLock NATIVE_CACHE_LOCK = new ReentrantReadWriteLock();

    private final String hostId = UUID.randomUUID().toString();
    private int affinityLimit = Integer.MAX_VALUE;
    private WorkerBinaryCache binaryCache;
    private Path taskRoot;
    private LlmTaskHandler llmTasks;
    private volatile ExecutorService pool;
    private volatile DeviceCapacityPinger capacityPinger;
    private final WorkerCapacity.Provider capacityProvider;
    private final Runnable nativeCapacityProbe;
    private final java.util.concurrent.atomic.AtomicInteger activeDagTasks = new java.util.concurrent.atomic.AtomicInteger();
    private volatile boolean stopped;
    private volatile DeviceOrchestratorClient orchestrator;
    private String scopedUrl;

    public ExecuteInlineWorker() { this(new WorkerCapacity.DesktopProvider()); }

    public ExecuteInlineWorker(WorkerCapacity.Provider capacityProvider) {
        this(capacityProvider, LlmSession::prepareCapacity);
    }

    ExecuteInlineWorker(WorkerCapacity.Provider capacityProvider, Runnable nativeCapacityProbe) {
        this.capacityProvider = Objects.requireNonNull(capacityProvider, "capacityProvider");
        this.nativeCapacityProbe = Objects.requireNonNull(nativeCapacityProbe, "nativeCapacityProbe");
    }

    /** Stops polling and in-flight transfers; native sessions close after the execution thread exits. */
    public void stop() {
        stopped = true;
        ExecutorService running = pool;
        if (running != null) running.shutdownNow();
        DeviceCapacityPinger sender = capacityPinger;
        if (sender != null) sender.close();
        DeviceOrchestratorClient client = orchestrator;
        if (client != null) client.close();
        if (binaryCache != null) binaryCache.close();
    }

    @Override
    public void execute(List<String> args) throws IOException {
        // args[0] = "worker", args[1] = optional url, args[2] = optional threads
        String environmentUrl = System.getenv("RAMANUJAN_WORKER_URL");
        String serverUrl = environmentUrl == null || environmentUrl.trim().isEmpty()
                ? "http://localhost:8888" : environmentUrl.trim().replaceAll("/+$", "");
        int numThreads = Runtime.getRuntime().availableProcessors();
        String cacheDir = Paths.get(System.getProperty("user.home"), ".ramanujan", "worker-cache").toString();
        boolean sharedFilesystem = false;
        int llmSessions = 8;
        Integer capacitySessionLimit = null;
        long diskReserveBytes = 64L << 20;

        List<String> positional = new ArrayList<>();
        for (int i = 1; i < args.size(); i++) {
            String arg = args.get(i).trim();
            switch (arg) {
                case "--native-library":
                    System.setProperty("ramanujan.nativeLibrary", requireValue(args, ++i, arg));
                    break;
                case "--cache":
                    cacheDir = requireValue(args, ++i, arg);
                    break;
                case "--max-shards":
                    affinityLimit = Integer.parseInt(requireValue(args, ++i, arg));
                    if (affinityLimit < 1) throw new IllegalArgumentException("--max-shards must be >= 1");
                    break;
                case "--shared-filesystem":
                    sharedFilesystem = true;
                    break;
                case "--llm-sessions":
                    llmSessions = Integer.parseInt(requireValue(args, ++i, arg));
                    if (llmSessions < 1) throw new IllegalArgumentException("--llm-sessions must be >= 1");
                    break;
                case "--disk-reserve-bytes":
                    diskReserveBytes = Long.parseLong(requireValue(args, ++i, arg));
                    if (diskReserveBytes < 0) throw new IllegalArgumentException("--disk-reserve-bytes must be >= 0");
                    break;
                case "--capacity-session-limit":
                    capacitySessionLimit = Integer.parseInt(requireValue(args, ++i, arg));
                    if (capacitySessionLimit < 1 || capacitySessionLimit > 1024)
                        throw new IllegalArgumentException("--capacity-session-limit must be 1..1024");
                    break;
                default:
                    if (arg.startsWith("--")) throw new IllegalArgumentException("unknown worker option " + arg);
                    positional.add(arg);
            }
        }
        if (!positional.isEmpty()) {
            String candidate = positional.get(0);
            if (candidate.startsWith("http")) {
                serverUrl = candidate.replaceAll("/$", "");
                if (positional.size() >= 2) {
                    try { numThreads = Integer.parseInt(positional.get(1)); }
                    catch (NumberFormatException ignored) {}
                }
            } else {
                try { numThreads = Integer.parseInt(candidate); }
                catch (NumberFormatException ignored) {}
            }
        }
        if (capacitySessionLimit != null && capacitySessionLimit < llmSessions)
            throw new IllegalArgumentException("--capacity-session-limit must be >= --llm-sessions");
        orchestrator = new DeviceOrchestratorClient(serverUrl);
        if (!sharedFilesystem) {
            binaryCache = new WorkerBinaryCache(Paths.get(cacheDir), serverUrl, diskReserveBytes);
            taskRoot = Paths.get(cacheDir).toAbsolutePath().resolve("tasks");
            Files.createDirectories(taskRoot);
        }
        llmTasks = new LlmTaskHandler(binaryCache, llmSessions,
                capacitySessionLimit == null ? llmSessions : capacitySessionLimit);

        System.out.println("LOCAL_WORKER_READY");
        scopedUrl = serverUrl.contains("/worker/") ? serverUrl : null;
        System.out.println("LOCAL_WORKER_URL " + (scopedUrl == null ? serverUrl : "[private room gateway]"));
        System.out.println("LOCAL_WORKER_THREADS " + numThreads);
        System.out.println("LOCAL_WORKER_HOST " + hostId);
        System.out.println("LOCAL_WORKER_BINARIES " + (sharedFilesystem ? "shared-filesystem" : "cache " + cacheDir));
        System.out.flush();

        final String url = serverUrl;
        final WorkerCapacity capacity = new WorkerCapacity(capacityProvider, Paths.get(cacheDir).toAbsolutePath(),
                hostId);
        capacityPinger = new DeviceCapacityPinger(url, () -> {
            Map<String, Object> snapshot = capacity.sample(llmTasks.openSessions(),
                    acceptingNewWork(), llmTasks.activePlans(), capacityDetails());
            snapshot.put("sessionLimit", llmTasks.sessionLimit());
            snapshot.put("capacitySessionLimit", llmTasks.capacitySessionLimit());
            snapshot.put("affinityLimit", affinityLimit == Integer.MAX_VALUE ? null : affinityLimit);
            snapshot.put("acceptingNewWork", acceptingNewWork());
            return snapshot;
        }, nativeCapacityProbe, error -> System.err.println("[Worker] capacity ping/preflight failed: " + safeError(error)));
        capacityPinger.start();
        pool = Executors.newFixedThreadPool(numThreads);
        for (int i = 0; i < numThreads; i++) {
            pool.submit(this::workerLoop);
        }

        pool.shutdown();
        try {
            pool.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);
        } catch (InterruptedException e) {
            stop();
            Thread.currentThread().interrupt();
        } finally {
            stop();
            // Native calls cannot be interrupted; do not close sessions while a call is using them.
            boolean interrupted = Thread.interrupted();
            while (!pool.isTerminated()) {
                try { pool.awaitTermination(1, TimeUnit.SECONDS); }
                catch (InterruptedException e) { interrupted = true; }
            }
            llmTasks.closeAll();
            if (interrupted) Thread.currentThread().interrupt();
        }
    }

    @SuppressWarnings("unchecked")
    private void workerLoop() {
        System.err.println("[Worker] started hostId=" + hostId);
        Integer pollAffinityLimit = affinityLimit == Integer.MAX_VALUE ? null : affinityLimit;

        while (!stopped && !Thread.currentThread().isInterrupted()) {
            boolean dagInFlight = false;
            try {
                long pollStarted = System.nanoTime();
                Map<String, Object> pingResp = orchestrator.pollTask(hostId, pollAffinityLimit);
                if (pingResp == null) {
                    idleBackoff(pollStarted);
                    continue;
                }
                if (!"SUCCESS".equalsIgnoreCase((String) pingResp.get("status"))) continue;

                Object dataObj = pingResp.get("data");
                if (dataObj == null) {
                    idleBackoff(pollStarted);
                    continue;
                }

                Map<String, Object> taskData = (Map<String, Object>) dataObj;
                if (taskData.get("llm") instanceof Map) {
                    runLlmTask((String) taskData.get("uuid"), (Map<String, Object>) taskData.get("llm"));
                    continue;
                }
                String uuid           = (String) taskData.get("uuid");
                String firstCommandId = (String) taskData.get("firstCommandId");
                Object reiObj         = taskData.get("ruleEngineInput");
                if (uuid == null || reiObj == null) continue;
                activeDagTasks.incrementAndGet();
                dagInFlight = true;

                System.err.println("[Worker] task " + uuid + " firstCmd=" + firstCommandId);

                // Deserialize as typed POJO so GPU flags are accessible
                RuleEngineInput rei = MAPPER.convertValue(reiObj, RuleEngineInput.class);

                List<String> gpuKernels = new ArrayList<>();
                if (rei.getFunctionCalls() != null) {
                    for (FunctionCall fc : rei.getFunctionCalls()) {
                        if (Boolean.TRUE.equals(fc.getIsGpu()) && fc.getId() != null) {
                            gpuKernels.add(fc.getId());
                        }
                    }
                }
                if (!gpuKernels.isEmpty()) {
                    System.err.println("[Worker] [GPU] " + gpuKernels.size()
                            + " GPU kernel(s): " + gpuKernels);
                }

                // Execute via NativeProcessor (same path as Android app)
                Map<String, Object> results = new HashMap<>();
                String error = null;
                boolean evictWeights = Boolean.TRUE.equals(taskData.get("evictWeights"));
                Path taskDir = taskRoot == null ? null : taskRoot.resolve(uuid);
                long start = System.currentTimeMillis();
                try {
                    if (binaryCache != null) {
                        binaryCache.localize(rei, taskDir);
                    }
                    NativeProcessor np = new NativeProcessor();
                    if (firstCommandId != null && !firstCommandId.isEmpty()) {
                        byte[] reiProto = RuleEngineInputProtoSerializer.serialize(rei);
                        NATIVE_CACHE_LOCK.readLock().lock();
                        try {
                            if (gpuKernels.isEmpty()) {
                                np.process(reiProto, firstCommandId);
                            } else {
                                synchronized (GPU_EXECUTION_LOCK) {
                                    np.process(reiProto, firstCommandId);
                                }
                            }
                        } finally {
                            NATIVE_CACHE_LOCK.readLock().unlock();
                        }
                        if (np.jniObject != null) results = np.jniObject;
                    }
                    if (evictWeights) {
                        NATIVE_CACHE_LOCK.writeLock().lock();
                        try {
                            np.changeShard();
                        } finally {
                            NATIVE_CACHE_LOCK.writeLock().unlock();
                        }
                    }
                } catch (Exception | LinkageError e) {
                    error = safeError(e);
                    System.err.println("[Worker] execution error for " + uuid + ": " + error);
                } finally {
                    deleteRecursively(taskDir);
                }
                long elapsed = System.currentTimeMillis() - start;

                if (!gpuKernels.isEmpty()) {
                    System.err.println("[Worker] [GPU] run latency: " + elapsed
                            + " ms (kernels: " + gpuKernels + ")");
                } else {
                    System.err.println("[Worker] task " + uuid + " done in " + elapsed + "ms"
                            + "  result keys=" + results.keySet());
                }

                // RETURN()-marked arrays too large for a JSON point-value map arrive here
                // as arrayId -> local temp file path (raw little-endian float32 bytes).
                // Upload each one's bytes directly, then strip the (worker-local, otherwise
                // meaningless) paths out of the JSON payload before reporting completion.
                Object binaryFilesObj = results.remove("binaryArrayFiles");
                if (binaryFilesObj instanceof Map) {
                    Map<String, String> binaryFiles = (Map<String, String>) binaryFilesObj;
                    for (Map.Entry<String, String> be : binaryFiles.entrySet()) {
                        String arrayId  = be.getKey();
                        String filePath = be.getValue();
                        try {
                            if (error == null) orchestrator.uploadBinary(uuid, arrayId, readAllBytes(filePath));
                        } catch (Exception uploadEx) {
                            error = "upload of binary array " + arrayId + " failed: " + safeError(uploadEx);
                            System.err.println("[Worker] " + error + " (task " + uuid + ")");
                        } finally {
                            new File(filePath).delete();
                        }
                    }
                }

                orchestrator.completeTask(uuid, hostId, results, error);

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Throwable t) {
                // Catch Throwable (not just Exception) so fatal errors like OutOfMemoryError
                // are logged instead of silently killing this thread: workerLoop() runs
                // inside pool.submit(), whose returned Future is never .get()'d, so an
                // uncaught Error here would otherwise vanish with no diagnostic output.
                System.err.println("[Worker] error: " + safeError(t));
                try { Thread.sleep(1000); }
                catch (InterruptedException e) { Thread.currentThread().interrupt(); break; }
            } finally {
                if (dagInFlight) activeDagTasks.decrementAndGet();
            }
        }
    }

    boolean acceptingNewWork() {
        return !stopped && activeDagTasks.get() == 0 && llmTasks != null && llmTasks.acceptingNewSessions();
    }

    /** Runs a native LLM stage step (or session close) and reports it to /task/complete. */
    private void runLlmTask(String uuid, Map<String, Object> task) throws Exception {
        Map<String, Object> results = new HashMap<>();
        String error = null;
        long start = System.currentTimeMillis();
        try {
            results = llmTasks.handle(task);
        } catch (Exception | LinkageError e) {
            error = safeError(e);
            System.err.println("[Worker] llm " + task.get("op") + " failed for session " + task.get("session") + ": " + error);
        }
        String what = task.get("stages") instanceof List
                ? "stages=" + ((List<?>) task.get("stages")).size() : "session=" + task.get("session");
        System.err.println("[Worker] llm " + task.get("op") + " " + what + " pos=" + task.get("pos")
                + " n=" + task.get("n") + " done in " + (System.currentTimeMillis() - start) + "ms"
                + " (open sessions " + llmTasks.openSessions() + ")");
        orchestrator.completeTask(uuid, hostId, results, error);
    }

    private static String requireValue(List<String> args, int index, String option) {
        if (index >= args.size()) throw new IllegalArgumentException(option + " requires a value");
        return args.get(index).trim();
    }

    private String safeError(Throwable error) {
        String message = error.getClass().getSimpleName() + ": " + error.getMessage();
        if (scopedUrl != null) message = message.replace(scopedUrl, "[private room gateway]");
        return message.replaceAll("(?i)/worker/[^\\s/?\"']+", "/worker/[redacted]");
    }

    static void idleBackoff(long pollStarted) throws InterruptedException {
        // Legacy central polling returns immediately; homelab's long-poll already supplies this delay.
        long remaining = TimeUnit.MILLISECONDS.toNanos(100) - (System.nanoTime() - pollStarted);
        if (remaining > 0) TimeUnit.NANOSECONDS.sleep(remaining);
    }

    private static void deleteRecursively(Path path) {
        if (path == null || !Files.exists(path)) return;
        try (java.util.stream.Stream<Path> walk = Files.walk(path)) {
            walk.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
        } catch (IOException e) {
            System.err.println("[Worker] could not delete " + path + ": " + e.getMessage());
        }
    }

    private static byte[] readAllBytes(String filePath) throws IOException {
        try (InputStream is = new FileInputStream(filePath)) {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] buf = new byte[65536];
            int n;
            while ((n = is.read(buf)) != -1) baos.write(buf, 0, n);
            return baos.toByteArray();
        }
    }

    private Map<String, Object> capacityDetails() {
        Map<String, Object> details = new LinkedHashMap<>();
        if (binaryCache == null) return details;
        WorkerBinaryCache.CacheInventory inventory = binaryCache.inventory();
        details.put("cachedFiles", inventory.files);
        details.put("cachedFilesTruncated", inventory.truncated);
        Map<String, Long> cached = new HashMap<>();
        for (Map<String, Object> file : inventory.files) {
            cached.put((String) file.get("path"), ((Number) file.get("bytes")).longValue());
        }
        details.put("cachedShards", binaryCache.shards().report(cached));
        return details;
    }
}

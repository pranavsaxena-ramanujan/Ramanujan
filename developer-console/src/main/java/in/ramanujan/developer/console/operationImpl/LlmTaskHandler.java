package in.ramanujan.developer.console.operationImpl;

import com.fasterxml.jackson.databind.ObjectMapper;
import in.ramanujan.rule.engine.LlmSession;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Executes homelab {@code /llm/step}, {@code /llm/chain} and {@code /llm/close} tasks on a worker.
 *
 * Each driver-side pipeline stage has a session key; the worker keeps one native
 * {@link LlmSession} per key (weights on the device, KV cache, recurrent state), so a
 * token step only carries token ids or one hidden vector per token. A chain task runs
 * several of this worker's stages back to back. Weight files named by the stage graph
 * are localized through the worker's persistent binary cache once.
 */
final class LlmTaskHandler {
    interface SessionFactory {
        Session open(String graphJson);
    }

    interface Session extends AutoCloseable {
        float[] step(int[] tokens, float[] hidden, int n, int pos);

        String info();

        @Override
        void close();
    }

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final WorkerBinaryCache binaryCache;
    private final int maxSessions;
    private final int capacityMaxSessions;
    private final SessionFactory factory;
    private final LinkedHashMap<String, Session> sessions = new LinkedHashMap<>();
    private final java.util.Set<String> opening = new java.util.HashSet<>();
    private final Map<String, String> sessionPlans = new LinkedHashMap<>();

    LlmTaskHandler(WorkerBinaryCache binaryCache, int maxSessions) {
        this(binaryCache, maxSessions, maxSessions);
    }

    LlmTaskHandler(WorkerBinaryCache binaryCache, int maxSessions, int capacityMaxSessions) {
        this(binaryCache, maxSessions, capacityMaxSessions, graph -> {
            final LlmSession session = new LlmSession(graph);
            return new Session() {
                public float[] step(int[] tokens, float[] hidden, int n, int pos) {
                    return session.step(tokens, hidden, n, pos);
                }

                public String info() {
                    return session.info();
                }

                public void close() {
                    session.close();
                }
            };
        });
    }

    LlmTaskHandler(WorkerBinaryCache binaryCache, int maxSessions, SessionFactory factory) {
        this(binaryCache, maxSessions, maxSessions, factory);
    }

    LlmTaskHandler(WorkerBinaryCache binaryCache, int maxSessions, int capacityMaxSessions, SessionFactory factory) {
        this.binaryCache = binaryCache;
        if (maxSessions < 1) throw new IllegalArgumentException("maxSessions must be >= 1");
        this.maxSessions = maxSessions;
        if (capacityMaxSessions < maxSessions) throw new IllegalArgumentException("capacity session limit must be >= maxSessions");
        this.capacityMaxSessions = capacityMaxSessions;
        this.factory = factory;
    }

    /** Runs one task; returns the result map posted to /task/complete. */
    @SuppressWarnings("unchecked")
    Map<String, Object> handle(Map<String, Object> task) throws IOException {
        String op = String.valueOf(task.get("op"));
        String key = String.valueOf(task.get("session"));
        Map<String, Object> result = new LinkedHashMap<>();
        if ("close".equals(op)) {
            Session session;
            synchronized (sessions) {
                if (opening.contains(key)) throw new IllegalStateException("LLM session " + key + " is opening; retry close");
                session = sessions.remove(key);
                sessionPlans.remove(key);
            }
            if (session != null) session.close();
            result.put("closed", session != null);
            return result;
        }
        if ("chain".equals(op)) return chain(task);
        if (!"step".equals(op)) throw new IllegalArgumentException("unknown llm op " + op);
        int n = ((Number) task.get("n")).intValue();
        int pos = ((Number) task.get("pos")).intValue();
        Session session = session(key, pos, (Map<String, Object>) task.get("graph"), planId(task));
        float[] hidden = task.get("hidden") != null ? decodeFloats(String.valueOf(task.get("hidden"))) : null;
        float[] out = session.step(tokens(task), hidden, n, pos);
        putOutput(result, out, task.get("output"));
        result.put("info", MAPPER.readValue(session.info(), Map.class));
        return result;
    }

    /**
     * Runs consecutive stages of one token back to back: {stages: [{session, graph?}], tokens | hidden, n,
     * pos, output}. Each stage's output feeds the next in memory, so the device never waits on the network
     * between them. Returns the last stage's output plus every stage's info, in order.
     */
    @SuppressWarnings("unchecked")
    private Map<String, Object> chain(Map<String, Object> task) throws IOException {
        int n = ((Number) task.get("n")).intValue();
        int pos = ((Number) task.get("pos")).intValue();
        if (binaryCache != null && pos == 0) {
            List<String> assigned = new ArrayList<>();
            synchronized (sessions) {
                for (Map<String, Object> stage : (List<Map<String, Object>>) task.get("stages")) {
                    if (!sessions.containsKey(String.valueOf(stage.get("session")))) collectFiles(stage.get("graph"), assigned);
                }
            }
            binaryCache.preflight(assigned);
        }
        int[] tokens = tokens(task);
        float[] hidden = task.get("hidden") != null ? decodeFloats(String.valueOf(task.get("hidden"))) : null;
        List<Object> infos = new ArrayList<>();
        Map<String, Object> info = null;
        for (Map<String, Object> stage : (List<Map<String, Object>>) task.get("stages")) {
            Session session = session(String.valueOf(stage.get("session")), pos, (Map<String, Object>) stage.get("graph"),
                    planId(task));
            hidden = session.step(tokens, hidden, n, pos);
            tokens = null;
            info = MAPPER.readValue(session.info(), Map.class);
            infos.add(info);
        }
        Map<String, Object> result = new LinkedHashMap<>();
        putOutput(result, hidden, task.get("output"));
        result.put("info", info);
        result.put("infos", infos);
        return result;
    }

    /** "argmax" returns only the chosen token id (greedy decoding); anything else returns all floats. */
    private static void putOutput(Map<String, Object> result, float[] out, Object mode) {
        if ("argmax".equals(mode)) {
            result.put("token", argmax(out));
        } else {
            result.put("output", encodeFloats(out));
        }
    }

    /** Lowest index on ties, matching numpy.argmax on the driver. */
    static int argmax(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) best = i;
        }
        return best;
    }

    @SuppressWarnings("unchecked")
    private static int[] tokens(Map<String, Object> task) {
        if (task.get("tokens") == null) return null;
        List<Number> ids = (List<Number>) task.get("tokens");
        int[] tokens = new int[ids.size()];
        for (int i = 0; i < tokens.length; i++) tokens[i] = ids.get(i).intValue();
        return tokens;
    }

    private static String planId(Map<String, Object> task) {
        Object id = task.get("planId");
        return id == null || String.valueOf(id).trim().isEmpty() ? null : String.valueOf(id);
    }

    @SuppressWarnings("unchecked")
    private Session session(String key, int pos, Map<String, Object> graph, String planId) throws IOException {
        synchronized (sessions) {
            Session existing = sessions.get(key);
            if (existing != null) return existing;
            if (opening.contains(key)) throw new IllegalStateException("LLM session " + key + " is opening; retry");
            if (pos != 0 || graph == null) {
                throw new IllegalStateException("LLM session " + key + " is not open on this worker (restarted or the "
                        + "shard moved); restart the generation from position 0");
            }
            int limit = planId == null ? maxSessions : capacityMaxSessions;
            if (sessions.size() + opening.size() >= limit) {
                throw new IllegalStateException("LLM session capacity reached (" + limit
                        + "); explicitly close an existing session before opening another");
            }
            opening.add(key);
            if (planId != null) sessionPlans.put(key, planId);
        }
        try {
            if (binaryCache != null) {
                List<String> weights = new ArrayList<>();
                collectFiles(graph, weights);
                binaryCache.preflight(weights);
            }
            Map<String, Object> original = binaryCache == null ? null : MAPPER.readValue(MAPPER.writeValueAsBytes(graph), Map.class);
            localize(graph);
            if (original != null) {
                try {
                    binaryCache.shards().record(original);
                } catch (IOException | RuntimeException e) {
                    System.err.println("[Worker] shard manifest not updated: " + e.getClass().getSimpleName());
                }
            }
            Session opened = factory.open(MAPPER.writeValueAsString(graph));
            synchronized (sessions) {
                sessions.put(key, opened);
                opening.remove(key);
            }
            return opened;
        } finally {
            synchronized (sessions) {
                opening.remove(key);
                if (!sessions.containsKey(key)) sessionPlans.remove(key);
            }
        }
    }

    /** Collects immutable tensor files without treating full weight bytes as an in-memory requirement. */
    private static void collectFiles(Object node, List<String> files) {
        if (node instanceof Map) {
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) node).entrySet()) {
                if ("file".equals(entry.getKey()) && entry.getValue() instanceof String) files.add((String) entry.getValue());
                else collectFiles(entry.getValue(), files);
            }
        } else if (node instanceof List) {
            for (Object item : (List<?>) node) collectFiles(item, files);
        }
    }

    /** Rewrites every tensor "file" in the graph to a worker-local cached copy. */
    @SuppressWarnings("unchecked")
    private void localize(Object node) throws IOException {
        if (binaryCache == null) return;
        if (node instanceof Map) {
            Map<String, Object> map = (Map<String, Object>) node;
            for (Map.Entry<String, Object> entry : map.entrySet()) {
                if ("file".equals(entry.getKey()) && entry.getValue() instanceof String) {
                    entry.setValue(binaryCache.cached((String) entry.getValue()).toString());
                } else {
                    localize(entry.getValue());
                }
            }
        } else if (node instanceof List) {
            for (Object item : (List<Object>) node) localize(item);
        }
    }

    int openSessions() {
        synchronized (sessions) {
            return sessions.size() + opening.size();
        }
    }

    boolean acceptingNewSessions() {
        synchronized (sessions) { return sessions.size() + opening.size() < capacityMaxSessions; }
    }

    int sessionLimit() { return maxSessions; }

    int capacitySessionLimit() { return capacityMaxSessions; }

    Map<String, Integer> activePlans() {
        synchronized (sessions) {
            Map<String, Integer> counts = new LinkedHashMap<>();
            for (String plan : sessionPlans.values()) counts.put(plan, counts.getOrDefault(plan, 0) + 1);
            return counts;
        }
    }

    void closeAll() {
        List<Session> all;
        synchronized (sessions) {
            all = new ArrayList<>(sessions.values());
            sessions.clear();
            sessionPlans.clear();
        }
        for (Session session : all) session.close();
    }

    static float[] decodeFloats(String base64) {
        ByteBuffer bytes = ByteBuffer.wrap(Base64.getDecoder().decode(base64)).order(ByteOrder.LITTLE_ENDIAN);
        float[] values = new float[bytes.remaining() / Float.BYTES];
        bytes.asFloatBuffer().get(values);
        return values;
    }

    static String encodeFloats(float[] values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        bytes.asFloatBuffer().put(values);
        return Base64.getEncoder().encodeToString(bytes.array());
    }
}

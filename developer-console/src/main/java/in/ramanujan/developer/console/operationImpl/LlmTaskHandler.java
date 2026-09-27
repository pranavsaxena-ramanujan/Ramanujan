package in.ramanujan.developer.console.operationImpl;

import com.fasterxml.jackson.databind.ObjectMapper;
import in.ramanujan.rule.engine.LlmSession;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Executes homelab {@code /llm/step} and {@code /llm/close} tasks on a worker.
 *
 * Each driver-side pipeline stage has a session key; the worker keeps one native
 * {@link LlmSession} per key (weights on the device, KV cache, recurrent state), so a
 * token step only carries token ids or one hidden vector per token. Weight files named
 * by the stage graph are localized through the worker's persistent binary cache once.
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
    private final SessionFactory factory;
    // Access-ordered: the least recently used session is closed when over the limit.
    private final LinkedHashMap<String, Session> sessions = new LinkedHashMap<>(16, 0.75f, true);

    LlmTaskHandler(WorkerBinaryCache binaryCache, int maxSessions) {
        this(binaryCache, maxSessions, graph -> {
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
        this.binaryCache = binaryCache;
        this.maxSessions = maxSessions;
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
                session = sessions.remove(key);
            }
            if (session != null) session.close();
            result.put("closed", session != null);
            return result;
        }
        if (!"step".equals(op)) throw new IllegalArgumentException("unknown llm op " + op);
        int n = ((Number) task.get("n")).intValue();
        int pos = ((Number) task.get("pos")).intValue();
        Session session = session(key, pos, (Map<String, Object>) task.get("graph"));
        int[] tokens = null;
        float[] hidden = null;
        if (task.get("tokens") != null) {
            List<Number> ids = (List<Number>) task.get("tokens");
            tokens = new int[ids.size()];
            for (int i = 0; i < tokens.length; i++) tokens[i] = ids.get(i).intValue();
        }
        if (task.get("hidden") != null) hidden = decodeFloats(String.valueOf(task.get("hidden")));
        float[] out = session.step(tokens, hidden, n, pos);
        result.put("output", encodeFloats(out));
        result.put("info", MAPPER.readValue(session.info(), Map.class));
        return result;
    }

    private Session session(String key, int pos, Map<String, Object> graph) throws IOException {
        synchronized (sessions) {
            Session existing = sessions.get(key);
            if (existing != null) return existing;
        }
        if (pos != 0 || graph == null) {
            throw new IllegalStateException("LLM session " + key + " is not open on this worker (restarted or the "
                    + "shard moved); restart the generation from position 0");
        }
        localize(graph);
        Session opened = factory.open(MAPPER.writeValueAsString(graph));
        List<Session> evicted = new ArrayList<>();
        synchronized (sessions) {
            Session raced = sessions.get(key);
            if (raced != null) {
                evicted.add(opened);
                opened = raced;
            } else {
                sessions.put(key, opened);
                Iterator<Map.Entry<String, Session>> oldest = sessions.entrySet().iterator();
                while (sessions.size() > maxSessions && oldest.hasNext()) {
                    Map.Entry<String, Session> entry = oldest.next();
                    if (entry.getKey().equals(key)) continue;
                    evicted.add(entry.getValue());
                    oldest.remove();
                }
            }
        }
        for (Session session : evicted) session.close();
        return opened;
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
            return sessions.size();
        }
    }

    void closeAll() {
        List<Session> all;
        synchronized (sessions) {
            all = new ArrayList<>(sessions.values());
            sessions.clear();
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

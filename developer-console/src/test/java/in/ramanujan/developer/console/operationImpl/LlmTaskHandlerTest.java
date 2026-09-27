package in.ramanujan.developer.console.operationImpl;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class LlmTaskHandlerTest {
    private final List<FakeSession> opened = new ArrayList<>();

    private final class FakeSession implements LlmTaskHandler.Session {
        final String graph;
        int position;
        boolean closed;

        FakeSession(String graph) {
            this.graph = graph;
        }

        public float[] step(int[] tokens, float[] hidden, int n, int pos) {
            if (pos != position) throw new IllegalStateException("pos " + pos + " != " + position);
            position += n;
            float[] out = new float[n];
            for (int i = 0; i < n; i++) out[i] = tokens != null ? tokens[i] : hidden[i] * 2;
            return out;
        }

        public String info() {
            return "{\"position\":" + position + "}";
        }

        public void close() {
            closed = true;
        }
    }

    private LlmTaskHandler handler(int maxSessions) {
        return new LlmTaskHandler(null, maxSessions, graph -> {
            FakeSession session = new FakeSession(graph);
            opened.add(session);
            return session;
        });
    }

    private static Map<String, Object> step(String session, int pos, Object graph, Object tokens, String hidden, int n) {
        Map<String, Object> task = new LinkedHashMap<>();
        task.put("op", "step");
        task.put("session", session);
        task.put("pos", pos);
        task.put("n", n);
        if (graph != null) task.put("graph", graph);
        if (tokens != null) task.put("tokens", tokens);
        if (hidden != null) task.put("hidden", hidden);
        return task;
    }

    private static Map<String, Object> graph(String file) {
        Map<String, Object> tensor = new HashMap<>();
        tensor.put("file", file);
        Map<String, Object> graph = new LinkedHashMap<>();
        graph.put("embed", tensor);
        return graph;
    }

    @Test
    public void sessionKeepsStateAcrossSteps() throws Exception {
        LlmTaskHandler handler = handler(4);
        Map<String, Object> first = handler.handle(step("s", 0, graph("/w.bin"), Arrays.asList(3, 4), null, 2));
        assertArrayEquals(new float[]{3, 4}, LlmTaskHandler.decodeFloats((String) first.get("output")), 0f);
        String hidden = LlmTaskHandler.encodeFloats(new float[]{1.5f});
        Map<String, Object> second = handler.handle(step("s", 2, null, null, hidden, 1));
        assertArrayEquals(new float[]{3f}, LlmTaskHandler.decodeFloats((String) second.get("output")), 0f);
        assertEquals(3, ((Map<?, ?>) second.get("info")).get("position"));
        assertEquals(1, opened.size());
        assertTrue(opened.get(0).graph.contains("/w.bin"));
    }

    @Test
    public void unknownSessionAfterPositionZeroIsAnError() throws Exception {
        LlmTaskHandler handler = handler(4);
        try {
            handler.handle(step("lost", 5, null, Arrays.asList(1), null, 1));
            fail("expected lost-session error");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains("not open on this worker"));
        }
        assertEquals(0, opened.size());
    }

    @Test
    public void leastRecentlyUsedSessionIsClosedOverTheLimit() throws Exception {
        LlmTaskHandler handler = handler(2);
        handler.handle(step("a", 0, graph("/a"), Arrays.asList(1), null, 1));
        handler.handle(step("b", 0, graph("/b"), Arrays.asList(1), null, 1));
        handler.handle(step("a", 1, null, Arrays.asList(1), null, 1));
        handler.handle(step("c", 0, graph("/c"), Arrays.asList(1), null, 1));
        assertEquals(2, handler.openSessions());
        assertTrue("b was least recently used", opened.get(1).closed);
        assertTrue(!opened.get(0).closed && !opened.get(2).closed);
    }

    @Test
    public void closeReleasesTheSession() throws Exception {
        LlmTaskHandler handler = handler(2);
        handler.handle(step("a", 0, graph("/a"), Arrays.asList(1), null, 1));
        Map<String, Object> close = new HashMap<>();
        close.put("op", "close");
        close.put("session", "a");
        assertEquals(true, handler.handle(close).get("closed"));
        assertEquals(false, handler.handle(close).get("closed"));
        assertTrue(opened.get(0).closed);
        assertEquals(0, handler.openSessions());
    }

    @Test
    public void floatsRoundTripAsLittleEndianBase64() {
        float[] values = {0f, -1.25f, 3.4e38f, Float.MIN_VALUE};
        assertArrayEquals(values, LlmTaskHandler.decodeFloats(LlmTaskHandler.encodeFloats(values)), 0f);
        assertEquals("AACAPw==", LlmTaskHandler.encodeFloats(new float[]{1f}));
    }
}

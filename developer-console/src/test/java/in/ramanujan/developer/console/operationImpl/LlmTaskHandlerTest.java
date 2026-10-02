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

    private static Map<String, Object> chain(int pos, Object tokens, String hidden, int n, String output, Object... stages) {
        Map<String, Object> task = new LinkedHashMap<>();
        task.put("op", "chain");
        List<Object> list = new ArrayList<>();
        for (int i = 0; i < stages.length; i += 2) {
            Map<String, Object> stage = new LinkedHashMap<>();
            stage.put("session", stages[i]);
            if (stages[i + 1] != null) stage.put("graph", stages[i + 1]);
            list.add(stage);
        }
        task.put("stages", list);
        task.put("pos", pos);
        task.put("n", n);
        if (tokens != null) task.put("tokens", tokens);
        if (hidden != null) task.put("hidden", hidden);
        if (output != null) task.put("output", output);
        return task;
    }

    @Test
    public void chainFeedsEachStageIntoTheNext() throws Exception {
        LlmTaskHandler handler = handler(4);
        Map<String, Object> first = handler.handle(chain(0, Arrays.asList(3, 4), null, 2, null,
                "a", graph("/a"), "b", graph("/b"), "c", graph("/c")));
        assertArrayEquals(new float[]{12, 16}, LlmTaskHandler.decodeFloats((String) first.get("output")), 0f);
        assertEquals(3, ((List<?>) first.get("infos")).size());
        assertEquals(3, opened.size());

        String hidden = LlmTaskHandler.encodeFloats(new float[]{1.5f});
        Map<String, Object> second = handler.handle(chain(2, null, hidden, 1, null, "b", null, "c", null));
        assertArrayEquals(new float[]{6f}, LlmTaskHandler.decodeFloats((String) second.get("output")), 0f);
        assertEquals(3, ((Map<?, ?>) second.get("info")).get("position"));
        assertEquals(2, opened.get(0).position);
    }

    @Test
    public void chainSessionsMustBeOpenAfterPositionZero() throws Exception {
        LlmTaskHandler handler = handler(4);
        handler.handle(chain(0, Arrays.asList(1), null, 1, null, "a", graph("/a")));
        try {
            handler.handle(chain(1, Arrays.asList(1), null, 1, null, "a", null, "lost", null));
            fail("expected lost-session error");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains("not open on this worker"));
        }
    }

    @Test
    public void argmaxReturnsOnlyTheChosenToken() throws Exception {
        LlmTaskHandler handler = handler(4);
        String hidden = LlmTaskHandler.encodeFloats(new float[]{1f, 5f, -2f, 5f});
        Map<String, Object> stepTask = step("s", 0, graph("/s"), null, hidden, 4);
        stepTask.put("output", "argmax");
        Map<String, Object> stepResult = handler.handle(stepTask);
        assertEquals(1, stepResult.get("token"));
        assertTrue(!stepResult.containsKey("output"));

        Map<String, Object> chained = handler.handle(chain(0, null, hidden, 4, "argmax", "a", graph("/a"), "b", graph("/b")));
        assertEquals(1, chained.get("token"));
        assertTrue(!chained.containsKey("output"));
        assertEquals(0, LlmTaskHandler.argmax(new float[]{Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY}));
    }
}

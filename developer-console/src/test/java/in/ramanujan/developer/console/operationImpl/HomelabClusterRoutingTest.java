package in.ramanujan.developer.console.operationImpl;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.lang.reflect.Field;
import java.util.*;
import java.util.concurrent.*;

import static org.junit.Assert.*;

public class HomelabClusterRoutingTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static Map<String, Object> map(Object... pairs) {
        Map<String, Object> result = new LinkedHashMap<>();
        for (int i = 0; i < pairs.length; i += 2) result.put((String) pairs[i], pairs[i + 1]);
        return result;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> request(String base, String route, Map<String, Object> body,
                                                int status) throws Exception {
        HttpURLConnection connection = (HttpURLConnection) new URL(base + route).openConnection();
        connection.setReadTimeout(6000);
        if (body != null) {
            connection.setRequestMethod("POST");
            connection.setDoOutput(true);
            connection.getOutputStream().write(MAPPER.writeValueAsBytes(body));
            connection.getOutputStream().close();
        }
        assertEquals(status, connection.getResponseCode());
        try (InputStream input = status < 400 ? connection.getInputStream() : connection.getErrorStream()) {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            byte[] bytes = new byte[4096];
            int n;
            while ((n = input.read(bytes)) >= 0) output.write(bytes, 0, n);
            return MAPPER.readValue(output.toByteArray(), Map.class);
        } finally {
            connection.disconnect();
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    public void llmAssignmentsSessionsFilesAndCompletionAreIsolated() throws Exception {
        ExecuteInlineHomelabServer homelab = new ExecuteInlineHomelabServer();
        ExecutorService runner = Executors.newSingleThreadExecutor();
        Path file = Files.createTempFile(Paths.get("."), "homelab-cluster-", ".bin");
        try {
            Files.write(file, new byte[]{1, 2, 3});
            homelab.startHttpServer(0);
            Field field = ExecuteInlineHomelabServer.class.getDeclaredField("server");
            field.setAccessible(true);
            String base = "http://localhost:" + ((HttpServer) field.get(homelab)).getAddress().getPort();
            request(base, "/pings/open?clusterId=A", null, 400);
            Future<Map<String, Object>> call = runner.submit(() -> request(base, "/llm/step",
                    map("clusterId", "A", "affinity", "shard", "session", "session",
                            "files", Arrays.asList(file.toString()), "timeout", 5), 200));
            assertNull(request(base, "/pings/open?uuid=b&clusterId=B", null, 200).get("data"));
            assertNull(request(base, "/pings/open?uuid=global", null, 200).get("data"));
            Map<String, Object> task = (Map<String, Object>) request(base,
                    "/pings/open?uuid=a&clusterId=A", null, 200).get("data");
            assertNotNull(task);
            assertEquals("A", task.get("clusterId"));
            Map<String, Object> llm = (Map<String, Object>) task.get("llm");
            assertNotEquals("session", llm.get("session"));
            String path = URLEncoder.encode(file.toString(), "UTF-8");
            request(base, "/binary/stat?path=" + path + "&uuid=b&clusterId=B", null, 403);
            request(base, "/binary/stat?path=" + path + "&uuid=global", null, 403);
            request(base, "/binary/stat?path=" + path + "&uuid=a&clusterId=A", null, 200);
            request(base, "/task/complete", map("uuid", task.get("uuid"), "hostId", "b",
                    "clusterId", "A", "data", map("token", 7)), 403);
            request(base, "/task/complete", map("uuid", task.get("uuid"), "hostId", "a",
                    "clusterId", "B", "data", map("token", 7)), 403);
            request(base, "/task/complete", map("uuid", task.get("uuid"), "hostId", "a",
                    "clusterId", "A", "data", map("token", 7)), 200);
            assertEquals(7, call.get(3, TimeUnit.SECONDS).get("token"));
            Future<Map<String, Object>> otherCall = runner.submit(() -> request(base, "/llm/step",
                    map("clusterId", "B", "affinity", "shard", "session", "session", "timeout", 5), 200));
            Map<String, Object> otherTask = (Map<String, Object>) request(base,
                    "/pings/open?uuid=b&clusterId=B", null, 200).get("data");
            assertNotNull(otherTask);
            assertNotEquals(llm.get("session"), ((Map<String, Object>) otherTask.get("llm")).get("session"));
            request(base, "/binary/stat?path=" + path + "&uuid=b&clusterId=B", null, 403);
            request(base, "/task/complete", map("uuid", otherTask.get("uuid"), "hostId", "b",
                    "clusterId", "B", "data", map("token", 8)), 200);
            assertEquals(8, otherCall.get(3, TimeUnit.SECONDS).get("token"));
            Future<Map<String, Object>> secondStage = runner.submit(() -> request(base, "/llm/step",
                    map("clusterId", "A", "affinity", "second", "session", "other", "timeout", 5), 200));
            Map<String, Object> secondTask = (Map<String, Object>) request(base,
                    "/pings/open?uuid=a&clusterId=A", null, 200).get("data");
            request(base, "/task/complete", map("uuid", secondTask.get("uuid"), "hostId", "a",
                    "clusterId", "A", "data", map("token", 1)), 200);
            secondStage.get(3, TimeUnit.SECONDS);
            Future<Map<String, Object>> chain = runner.submit(() -> request(base, "/llm/chain",
                    map("clusterId", "A", "stages", Arrays.asList(
                            map("affinity", "shard", "session", "session", "files", Arrays.asList(file.toString())),
                            map("affinity", "second", "session", "other", "files", Arrays.asList(file.toString()))),
                            "timeout", 5), 200));
            Map<String, Object> chainTask = (Map<String, Object>) request(base,
                    "/pings/open?uuid=a&clusterId=A", null, 200).get("data");
            assertEquals("A", chainTask.get("clusterId"));
            assertEquals(2, ((List<?>) ((Map<?, ?>) chainTask.get("llm")).get("stages")).size());
            request(base, "/binary/stat?path=" + path + "&uuid=a&clusterId=A", null, 200);
            request(base, "/task/complete", map("uuid", chainTask.get("uuid"), "hostId", "a",
                    "clusterId", "A", "data", map("token", 9)), 200);
            Map<String, Object> chainResult = chain.get(3, TimeUnit.SECONDS);
            assertEquals(1, chainResult.get("tasks"));
            assertEquals(9, chainResult.get("token"));
        } finally {
            runner.shutdownNow();
            homelab.stopHttpServer();
            Files.deleteIfExists(file);
        }
    }

    @Test
    public void resultLookupRequiresSameClusterAndExplicitRequestId() throws Exception {
        ExecuteInlineHomelabServer homelab = new ExecuteInlineHomelabServer();
        Map<String, Map<String, Object>> arrays = new HashMap<>();
        arrays.put("x", map("0", 42));
        homelab.completeRunState("request", Collections.emptyMap(), arrays,
                Collections.emptyMap(), Collections.emptyMap(), "A");
        Path output = Files.createTempFile(Paths.get("."), "homelab-result-", ".csv");
        try {
            homelab.startHttpServer(0);
            Field field = ExecuteInlineHomelabServer.class.getDeclaredField("server");
            field.setAccessible(true);
            String base = "http://localhost:" + ((HttpServer) field.get(homelab)).getAddress().getPort();
            request(base, "/orchestrator/run", map("code", "x = 1", "clusterId", "A",
                    "csvInformationList", Arrays.asList(map("fileName", "/server/private.csv", "data", "1"))), 400);
            request(base, "/orchestrator/dump?clusterId=B", map("requestId", "request", "name", "x"), 404);
            request(base, "/orchestrator/dump", map("requestId", "request", "name", "x"), 404);
            request(base, "/orchestrator/dump?clusterId=A", map("name", "x"), 404);
            request(base, "/orchestrator/dump?clusterId=A",
                    map("requestId", "request", "name", "x", "path", output.toString()), 200);
            assertEquals("42\n", new String(Files.readAllBytes(output), "UTF-8"));
        } finally {
            homelab.stopHttpServer();
            Files.deleteIfExists(output);
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    public void nativeDagSuccessorsKeepClusterTag() throws Exception {
        ExecuteInlineHomelabServer homelab = new ExecuteInlineHomelabServer();
        ExecutorService runner = Executors.newSingleThreadExecutor();
        Path kernel = Files.createTempFile(Paths.get("."), "homelab-dag-", ".txt");
        try {
            Files.write(kernel, ("var result[1] : array;\n"
                    + "threadStart(t0) {\nzero = 0;\nresult[zero] = 7.0;\n}\n").getBytes("UTF-8"));
            homelab.startHttpServer(0);
            Field field = ExecuteInlineHomelabServer.class.getDeclaredField("server");
            field.setAccessible(true);
            String base = "http://localhost:" + ((HttpServer) field.get(homelab)).getAddress().getPort();
            List<Map<String, Object>> submissions = Arrays.asList(
                    map("clusterId", "A", "requestId", "legacy-dag", "args", Arrays.asList(kernel.toString())),
                    map("clusterId", "A", "requestId", "inline-dag", "code",
                            new String(Files.readAllBytes(kernel), "UTF-8"), "csvInformationList", Collections.emptyList()));
            for (Map<String, Object> submission : submissions) {
                Future<Map<String, Object>> call = runner.submit(() -> request(base, "/orchestrator/run", submission, 200));
                assertNull(request(base, "/pings/open?uuid=b&clusterId=B", null, 200).get("data"));
                int dispatched = 0;
                for (int i = 0; i < 10 && !call.isDone(); i++) {
                    Map<String, Object> task = (Map<String, Object>) request(base,
                            "/pings/open?uuid=a&clusterId=A", null, 200).get("data");
                    if (task == null) continue;
                    assertEquals("A", task.get("clusterId"));
                    dispatched++;
                    request(base, "/orchestrator/uploadBinary?taskUuid=" + task.get("uuid")
                                    + "&uuid=b&clusterId=A&arrayId=unknown", Collections.emptyMap(), 403);
                    request(base, "/orchestrator/uploadBinary?taskUuid=" + task.get("uuid")
                                    + "&uuid=a&clusterId=B&arrayId=unknown", Collections.emptyMap(), 403);
                    request(base, "/task/complete", map("uuid", task.get("uuid"), "hostId", "a",
                            "clusterId", "A", "data", Collections.emptyMap()), 200);
                }
                assertEquals(submission.get("requestId"), call.get(3, TimeUnit.SECONDS).get("requestId"));
                assertTrue(dispatched > 0);
                assertEquals("Array not found: x", request(base, "/orchestrator/dump?clusterId=A",
                        map("requestId", submission.get("requestId"), "name", "x"), 404).get("message"));
                assertEquals("No completed run state found", request(base, "/orchestrator/dump?clusterId=B",
                        map("requestId", submission.get("requestId"), "name", "x"), 404).get("message"));
            }
        } finally {
            runner.shutdownNow();
            homelab.stopHttpServer();
            Files.deleteIfExists(kernel);
        }
    }
}

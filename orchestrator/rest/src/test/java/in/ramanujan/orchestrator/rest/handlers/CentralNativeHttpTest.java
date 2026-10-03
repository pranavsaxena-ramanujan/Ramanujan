package in.ramanujan.orchestrator.rest.handlers;

import com.fasterxml.jackson.databind.ObjectMapper;
import in.ramanujan.db.layer.utils.QueryExecutor;
import in.ramanujan.orchestrator.base.pojo.AsyncTask;
import in.ramanujan.orchestrator.data.dao.*;
import in.ramanujan.orchestrator.data.impl.asyncTaskDaoImpl.AsyncTaskDaoSqlImpl;
import in.ramanujan.orchestrator.data.impl.asyncTaskHostMappingDaoImpl.AsyncTaskMappingSqlDbImpl;
import in.ramanujan.orchestrator.data.impl.hostDaoImpl.HostDaoStackImpl;
import in.ramanujan.orchestrator.service.*;
import in.ramanujan.pojo.RuleEngineInput;
import in.ramanujan.pojo.ruleEngineInputUnitsExt.array.Array;
import io.vertx.core.Vertx;
import io.vertx.core.http.HttpServer;
import io.vertx.ext.web.Router;
import io.vertx.ext.web.handler.BodyHandler;
import org.junit.Test;
import org.mockito.ArgumentCaptor;

import java.io.*;
import java.lang.reflect.Field;
import java.net.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;

import static org.junit.Assert.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

public class CentralNativeHttpTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static void inject(Object target, String name, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(name);
        field.setAccessible(true);
        field.set(target, value);
    }

    private static Map<String, Object> map(Object... pairs) {
        Map<String, Object> map = new LinkedHashMap<>();
        for (int i = 0; i < pairs.length; i += 2) map.put((String) pairs[i], pairs[i + 1]);
        return map;
    }

    private static final class Fixture implements AutoCloseable {
        final Vertx vertx = Vertx.vertx();
        final AsyncTaskDaoSqlImpl tasks = new AsyncTaskDaoSqlImpl();
        final AsyncTaskMappingSqlDbImpl mappings = new AsyncTaskMappingSqlDbImpl();
        final HostDaoStackImpl hosts = new HostDaoStackImpl();
        final StorageDao storage = mock(StorageDao.class);
        final ExecutorService requests = Executors.newFixedThreadPool(2);
        final HttpServer server;
        final String url;

        Fixture() throws Exception {
            QueryExecutor query = new QueryExecutor();
            query.init(null, QueryExecutor.DB_TYPE.IN_MEM, null);
            inject(tasks, "queryExecutor", query);
            NativeAffinityDao affinities = new NativeAffinityDao();
            inject(affinities, "queryExecutor", query);
            inject(affinities, "asyncTaskDao", tasks);
            NativeDeliveryDao deliveries = new NativeDeliveryDao();
            inject(deliveries, "queryExecutor", query);
            inject(mappings, "queryExecutor", query);
            inject(mappings, "asyncTaskDao", tasks);
            inject(mappings, "nativeAffinityDao", affinities);
            hosts.init();
            inject(hosts, "asyncTaskHostMappingDao", mappings);
            inject(hosts, "nativeAffinityDao", affinities);
            inject(hosts, "nativeDeliveryDao", deliveries);
            inject(hosts, "storageDao", storage);
            when(storage.getBreakpoints(anyString())).thenReturn(io.vertx.core.Future.succeededFuture());
            when(storage.storeAsyncTaskResult(anyString(), any())).thenReturn(io.vertx.core.Future.succeededFuture());
            NativeLlmService nativeService = new NativeLlmService();
            inject(nativeService, "asyncTaskDao", tasks);
            inject(nativeService, "hostMappingDao", mappings);
            inject(nativeService, "hostsDao", hosts);
            inject(nativeService, "affinityDao", affinities);
            NativeLlmHandler nativeHandler = new NativeLlmHandler();
            inject(nativeHandler, "service", nativeService);
            OpenPingService pingService = new OpenPingService();
            inject(pingService, "hostsDao", hosts);
            OpenPingHandler ping = new OpenPingHandler();
            inject(ping, "openPingService", pingService);
            TaskCompleteService completeService = new TaskCompleteService();
            inject(completeService, "asyncTaskHostMappingDao", mappings);
            inject(completeService, "asyncTaskDao", tasks);
            inject(completeService, "nativeLlmService", nativeService);
            inject(completeService, "storageDao", storage);
            TaskCompleteHandler complete = new TaskCompleteHandler();
            inject(complete, "taskCompleteService", completeService);
            WorkerBinaryHandler binary = new WorkerBinaryHandler();
            inject(binary, "mappings", mappings);
            inject(binary, "storage", storage);
            WorkerBinaryUploadHandler upload = new WorkerBinaryUploadHandler();
            inject(upload, "mappings", mappings);
            inject(upload, "tasks", tasks);
            inject(upload, "storage", storage);
            Router router = Router.router(vertx);
            router.route().handler(BodyHandler.create());
            router.post("/llm/step").handler(event -> nativeHandler.handle(event, "step"));
            router.post("/llm/chain").handler(event -> nativeHandler.handle(event, "chain"));
            router.post("/llm/close").handler(event -> nativeHandler.handle(event, "close"));
            router.post("/pings/open").handler(ping);
            router.post("/task/complete").handler(complete);
            router.get("/binary/stat").handler(event -> binary.handle(event, true));
            router.get("/binary/fetch").handler(event -> binary.handle(event, false));
            router.post("/orchestrator/uploadBinary").handler(upload::handle);
            CompletableFuture<HttpServer> ready = new CompletableFuture<>();
            vertx.createHttpServer().requestHandler(router).listen(0, "127.0.0.1", listening -> {
                if (listening.succeeded()) ready.complete(listening.result());
                else ready.completeExceptionally(listening.cause());
            });
            server = ready.get(5, TimeUnit.SECONDS);
            url = "http://127.0.0.1:" + server.actualPort();
        }

        byte[] raw(String path, byte[] body, int status) throws Exception {
            HttpURLConnection connection = (HttpURLConnection) new URL(url + path).openConnection();
            connection.setReadTimeout(6000);
            if (body != null) {
                connection.setRequestMethod("POST");
                connection.setDoOutput(true);
                connection.setRequestProperty("Content-Type", path.contains("uploadBinary") ? "application/octet-stream" : "application/json");
                try (OutputStream output = connection.getOutputStream()) { output.write(body); }
            }
            assertEquals(path, status, connection.getResponseCode());
            try (InputStream input = status < 400 ? connection.getInputStream() : connection.getErrorStream();
                 ByteArrayOutputStream output = new ByteArrayOutputStream()) {
                byte[] bytes = new byte[4096];
                int count;
                while ((count = input.read(bytes)) >= 0) output.write(bytes, 0, count);
                return output.toByteArray();
            } finally { connection.disconnect(); }
        }

        @SuppressWarnings("unchecked")
        Map<String, Object> json(String path, Map<String, Object> body, int status) throws Exception {
            return MAPPER.readValue(raw(path, body == null ? null : MAPPER.writeValueAsBytes(body), status), Map.class);
        }

        @SuppressWarnings("unchecked")
        Map<String, Object> poll(String host, String cluster) throws Exception {
            long deadline = System.currentTimeMillis() + 4000;
            while (System.currentTimeMillis() < deadline) {
                Object task = json("/pings/open?uuid=" + host + "&clusterId=" + cluster, Collections.emptyMap(), 200).get("data");
                if (task instanceof Map) return (Map<String, Object>) task;
                Thread.sleep(20);
            }
            throw new AssertionError("No task for device " + host);
        }

        @Override public void close() throws Exception {
            requests.shutdownNow();
            CountDownLatch closed = new CountDownLatch(1);
            server.close(done -> vertx.close(stopped -> closed.countDown()));
            assertTrue(closed.await(5, TimeUnit.SECONDS));
        }
    }

    @Test
    public void nativeStepUsesCentralPollCompletionAndClusterScopedBinaryRoutes() throws Exception {
        Path file = Files.createTempFile(Paths.get("."), "central_native_", ".bin");
        try (Fixture f = new Fixture()) {
            byte[] bytes = new byte[]{1, 2, 3, 4};
            Files.write(file, bytes);
            Future<Map<String, Object>> roomA = f.requests.submit(() -> f.json("/llm/step",
                    map("clusterId", "A", "affinity", "shard", "session", "chat",
                            "files", Arrays.asList(file.toString()), "timeout", 5), 200));
            Map<String, Object> taskA = f.poll("a", "A");
            assertEquals("A", taskA.get("clusterId"));
            Future<Map<String, Object>> roomB = f.requests.submit(() -> f.json("/llm/step",
                    map("clusterId", "B", "affinity", "shard", "session", "chat", "timeout", 5), 200));
            Map<String, Object> taskB = f.poll("b", "B");
            String path = URLEncoder.encode(file.toString(), "UTF-8");
            f.json("/binary/stat?path=" + path + "&uuid=b&clusterId=B", null, 403);
            f.json("/binary/stat?path=" + path + "&uuid=a&clusterId=B", null, 403);
            assertEquals(4, ((Number) f.json("/binary/stat?path=" + path + "&uuid=a&clusterId=A", null, 200).get("size")).intValue());
            assertArrayEquals(bytes, f.raw("/binary/fetch?path=" + path + "&uuid=a&clusterId=A", null, 200));
            f.json("/task/complete", map("uuid", taskA.get("uuid"), "hostId", "b", "clusterId", "A", "data", map("token", 1)), 403);
            f.json("/task/complete", map("uuid", taskA.get("uuid"), "hostId", "a", "clusterId", "A", "data", map("token", 7)), 200);
            f.json("/task/complete", map("uuid", taskB.get("uuid"), "hostId", "b", "clusterId", "B", "data", map("token", 8)), 200);
            assertEquals(7, roomA.get(3, TimeUnit.SECONDS).get("token"));
            assertEquals(8, roomB.get(3, TimeUnit.SECONDS).get("token"));
            assertEquals("COMPLETED", f.tasks.getAsyncTask((String) taskA.get("uuid")).result().getNativeState());
            assertNull(f.json("/pings/open?uuid=a&clusterId=A", Collections.emptyMap(), 200).get("data"));
        } finally { Files.deleteIfExists(file); }
    }

    @Test
    @SuppressWarnings("unchecked")
    public void taggedDagBinaryUploadIsAssignmentCheckedAndMergedIntoLegacyResultStorage() throws Exception {
        List<String> files = new ArrayList<>();
        try (Fixture f = new Fixture()) {
            RuleEngineInput input = new RuleEngineInput();
            Array array = new Array();
            array.setId("result-array");
            array.setDimension(Arrays.asList(2));
            input.getArrays().add(array);
            when(f.storage.getAsyncTaskRuleEngineInput("dag")).thenReturn(io.vertx.core.Future.succeededFuture(input));
            AsyncTask task = new AsyncTask();
            task.setUuid("dag");
            task.setClusterId("A");
            task.setStatus("PROCESSING");
            task.setDebug(false);
            task.setFirstCommandId("command");
            assertTrue(f.tasks.insert(task).succeeded());
            f.hosts.putMachineForComputation("a", "A");
            assertEquals("a", f.hosts.getMachine(task, false).result());
            assertNotNull(f.poll("a", "A"));
            assertNull(f.json("/pings/open?uuid=a&clusterId=A", Collections.emptyMap(), 200).get("data"));
            byte[] values = ByteBuffer.allocate(2 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN)
                    .putFloat(1.25f).putFloat(2.5f).array();
            f.raw("/orchestrator/uploadBinary?taskUuid=dag&uuid=b&clusterId=A&arrayId=result-array", values, 403);
            f.raw("/orchestrator/uploadBinary?taskUuid=dag&uuid=a&clusterId=B&arrayId=result-array", values, 403);
            f.raw("/orchestrator/uploadBinary?taskUuid=dag&uuid=a&clusterId=A&arrayId=result-array", values, 200);
            files.addAll(f.tasks.getAsyncTask("dag").result().getBinaryArrayFiles().values());
            f.json("/task/complete", map("uuid", "dag", "hostId", "a", "clusterId", "A", "data", Collections.emptyMap()), 200);
            ArgumentCaptor<Object> result = ArgumentCaptor.forClass(Object.class);
            verify(f.storage).storeAsyncTaskResult(eq("dag"), result.capture());
            Map<String, Object> stored = MAPPER.readValue(result.getValue().toString(), Map.class);
            Map<String, Object> resultMap = (Map<String, Object>) stored.get("map");
            Map<String, Object> arrays = (Map<String, Object>) resultMap.get("arrayIndex");
            Map<String, Object> output = (Map<String, Object>) arrays.get("result-array");
            assertEquals(1.25, ((Number) output.get("0")).doubleValue(), 0.0);
            assertEquals(2.5, ((Number) output.get("1")).doubleValue(), 0.0);
            assertEquals("SUCCESS", f.tasks.getAsyncTask("dag").result().getStatus());
        } finally {
            for (String file : files) Files.deleteIfExists(Paths.get(file));
            Path directory = Paths.get(".ramanujan-central-outputs");
            if (Files.isDirectory(directory)) {
                try (DirectoryStream<Path> contents = Files.newDirectoryStream(directory)) {
                    if (!contents.iterator().hasNext()) Files.delete(directory);
                }
            }
        }
    }
}

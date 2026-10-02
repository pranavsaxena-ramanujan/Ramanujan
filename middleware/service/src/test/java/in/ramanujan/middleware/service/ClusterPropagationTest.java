package in.ramanujan.middleware.service;

import in.ramanujan.data.KafkaManagerApiCaller;
import in.ramanujan.data.OrchestrationApiCaller;
import in.ramanujan.data.db.dao.*;
import in.ramanujan.data.db.impl.DagElementDao.DagElementSqlDbImpl;
import in.ramanujan.data.db.impl.asyncTaskDao.AsyncTaskHashMapImpl;
import in.ramanujan.data.db.impl.asyncTaskDao.AsyncTaskSqlDbImpl;
import in.ramanujan.db.layer.enums.StorageType;
import in.ramanujan.db.layer.utils.QueryExecutor;
import in.ramanujan.middleware.base.pojo.asyncTask.AsyncTask;
import in.ramanujan.pojo.RuleEngineInput;
import in.ramanujan.translation.codeConverter.BasicDagElement;
import in.ramanujan.translation.codeConverter.DagElement;
import in.ramanujan.translation.codeConverter.pojo.TranslateResponse;
import in.ramanujan.translation.codeConverter.pojo.VariableAndArrayResult;
import io.vertx.core.Future;
import io.vertx.core.Vertx;
import org.junit.Test;

import java.lang.reflect.Field;
import java.util.*;
import java.util.concurrent.*;

import static org.junit.Assert.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

public class ClusterPropagationTest {
    private static void inject(Object target, String fieldName, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }

    private static <T> T await(Future<T> future) throws Exception {
        CompletableFuture<T> completed = new CompletableFuture<>();
        future.setHandler(result -> {
            if (result.succeeded()) completed.complete(result.result());
            else completed.completeExceptionally(result.cause());
        });
        return completed.get(5, TimeUnit.SECONDS);
    }

    private static final class MemoryStorage extends StorageDao {
        private final Map<String, String> values = new ConcurrentHashMap<>();

        @Override
        protected void setObject(String id, String bucket, String value, int retry) {
            values.put(bucket + ":" + id, value);
        }

        @Override
        protected String getObject(String id, String bucket, int retry) {
            return values.get(bucket + ":" + id);
        }
    }

    @Test
    public void submissionStorageSuccessorsAndSubmissionRetriesPreserveCluster() throws Exception {
        Vertx vertx = Vertx.vertx();
        try {
            MemoryStorage storage = new MemoryStorage();
            storage.setContext(vertx.getOrCreateContext(), StorageType.LOCAL);
            DagElement first = new DagElement(new RuleEngineInput());
            first.setFirstCommandId("first-command");
            DagElement next = new DagElement(new RuleEngineInput());
            next.setFirstCommandId("next-command");
            first.getNextElements().add(next);
            next.getPreviousElements().add(first);
            TranslateResponse translated = new TranslateResponse();
            translated.setFirstDagElement(first);
            translated.setDagElementList(Arrays.asList(first, next));
            translated.setCodeAndDagElementMap(new HashMap<>());
            translated.getCodeAndDagElementMap().put(first.getId(), "first");
            translated.getCodeAndDagElementMap().put(next.getId(), "next");
            translated.setCommonFunctionCode("");
            AsyncTaskHashMapImpl asyncDao = new AsyncTaskHashMapImpl();
            asyncDao.init();
            DagElementDao dagDao = mock(DagElementDao.class);
            when(dagDao.mapDagElementToAsyncId(anyString(), anyList())).thenReturn(Future.succeededFuture());
            when(dagDao.addDagElementDependencies(anyString(), anyList())).thenReturn(Future.succeededFuture());
            when(dagDao.setDagElementAndOrchestratorAsyncIdMapping(anyString(), anyString())).thenReturn(Future.succeededFuture());
            KafkaManagerApiCaller kafka = mock(KafkaManagerApiCaller.class);
            when(kafka.callEventApi(anyString(), anyString(), eq(false))).thenReturn(Future.succeededFuture());
            OrchestrationApiCaller orchestrator = mock(OrchestrationApiCaller.class);
            when(orchestrator.runCode(anyString(), anyString(), anyString(), eq(vertx), anyString(),
                    eq(false), nullable(String.class), eq("A")))
                    .thenReturn(Future.failedFuture("retry"), Future.succeededFuture(Collections.emptyMap()),
                            Future.succeededFuture(Collections.emptyMap()));
            RunService service = new RunService();
            inject(service, "asyncTaskDao", asyncDao);
            inject(service, "dagElementDao", dagDao);
            inject(service, "storageDao", storage);
            inject(service, "kafkaManagerApiCaller", kafka);
            inject(service, "orchestrationApiCaller", orchestrator);
            String request = await(service.runCode(translated, vertx, false, "A"));
            assertEquals("A", asyncDao.getAsyncTask(request).result().getClusterId());
            assertEquals("A", first.getClusterId());
            assertEquals("A", next.getClusterId());
            assertEquals("A", await(storage.getDagElement(next.getId())).getClusterId());
            await(service.runDagElementId(request, next.getId(), vertx, false));
            verify(orchestrator, times(2)).runCode(eq(request), eq(first.getFirstCommandId()), eq(first.getId()),
                    eq(vertx), anyString(), eq(false), nullable(String.class), eq("A"));
            verify(orchestrator).runCode(eq(request), eq(next.getFirstCommandId()), eq(next.getId()),
                    eq(vertx), anyString(), eq(false), nullable(String.class), eq("A"));
        } finally {
            CountDownLatch closed = new CountDownLatch(1);
            vertx.close(result -> closed.countDown());
            assertTrue(closed.await(5, TimeUnit.SECONDS));
        }
    }

    @Test
    public void sqlDagAndAsyncRowsRoundTripClusterAcrossUpdates() throws Exception {
        QueryExecutor query = new QueryExecutor();
        query.init(null, QueryExecutor.DB_TYPE.IN_MEM, null);
        DagElementSqlDbImpl dags = new DagElementSqlDbImpl();
        inject(dags, "queryExecutor", query);
        DagElement dag = new DagElement(new RuleEngineInput());
        dag.setClusterId("A");
        dag.setFirstCommandId("command");
        await(dags.addElement(dag));
        BasicDagElement loaded = await(dags.getDagElement(dag.getId()));
        assertEquals("A", loaded.getClusterId());
        assertEquals("command", loaded.getFirstCommandId());
        AsyncTaskSqlDbImpl async = new AsyncTaskSqlDbImpl();
        inject(async, "queryExecutor", query);
        AsyncTask task = new AsyncTask();
        task.setTaskId("request");
        task.setTaskStatus(AsyncTask.TaskStatus.PENDING);
        task.setClusterId("A");
        await(async.insert(task));
        await(async.update("request", Collections.singletonMap("taskStatus", AsyncTask.TaskStatus.SUCCESS)));
        assertEquals("A", await(async.getAsyncTask("request")).getClusterId());
        assertEquals(AsyncTask.TaskStatus.SUCCESS, await(async.getAsyncTask("request")).getTaskStatus());
    }

    @Test
    public void completedResultsPreserveClusterAndRejectOtherRoomOrGlobalLookup() throws Exception {
        AsyncTaskHashMapImpl async = new AsyncTaskHashMapImpl();
        async.init();
        AsyncTask task = new AsyncTask();
        task.setTaskId("request");
        task.setTaskStatus(AsyncTask.TaskStatus.PENDING);
        task.setClusterId("A");
        async.insert(task);
        TaskStatusService service = new TaskStatusService();
        inject(service, "asyncTaskDao", async);
        DagElementDao dags = mock(DagElementDao.class);
        when(dags.isAsyncTaskDone("request")).thenReturn(Future.succeededFuture(0));
        inject(service, "dagElementDao", dags);
        VariableValueDao values = mock(VariableValueDao.class);
        when(values.getAllValuesForAsyncId("request")).thenReturn(Future.succeededFuture(
                new VariableAndArrayResult(Collections.emptyList(), Collections.emptyList())));
        inject(service, "variableValueDao", values);
        assertTrue(service.getAsyncTaskStatus("request", "B").cause() instanceof SecurityException);
        assertTrue(service.getAsyncTaskStatus("request").cause() instanceof SecurityException);
        verifyZeroInteractions(values);
        AsyncTask completed = await(service.getAsyncTaskStatus("request", "A"));
        assertEquals("A", completed.getClusterId());
        assertEquals(AsyncTask.TaskStatus.SUCCESS, completed.getTaskStatus());
    }
}

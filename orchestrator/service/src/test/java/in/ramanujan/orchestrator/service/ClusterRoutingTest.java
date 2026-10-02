package in.ramanujan.orchestrator.service;

import in.ramanujan.db.layer.utils.QueryExecutor;
import in.ramanujan.orchestrator.base.enums.Status;
import in.ramanujan.orchestrator.base.pojo.AsyncTask;
import in.ramanujan.orchestrator.base.pojo.CheckpointResumePayload;
import in.ramanujan.orchestrator.data.dao.*;
import in.ramanujan.orchestrator.data.external.OrchestratorApiCaller;
import in.ramanujan.orchestrator.data.impl.asyncTaskDaoImpl.AsyncTaskDaoSqlImpl;
import in.ramanujan.orchestrator.data.impl.asyncTaskHostMappingDaoImpl.AsyncTaskHostMappingDaoHashMapImpl;
import in.ramanujan.orchestrator.data.impl.asyncTaskHostMappingDaoImpl.AsyncTaskMappingSqlDbImpl;
import in.ramanujan.orchestrator.data.impl.hostDaoImpl.HostDaoStackImpl;
import in.ramanujan.pojo.RuleEngineInput;
import io.vertx.core.Future;
import org.junit.Test;
import org.mockito.ArgumentCaptor;

import java.lang.reflect.Field;
import java.util.*;

import static org.junit.Assert.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

public class ClusterRoutingTest {
    private static void inject(Object target, String fieldName, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }

    private static AsyncTask task(String id, String cluster) {
        AsyncTask task = new AsyncTask();
        task.setUuid(id);
        task.setClusterId(cluster);
        task.setStatus(Status.PROCESSING.getKeyName());
        task.setDebug(false);
        return task;
    }

    private static HostDaoStackImpl hosts(AsyncTaskHostMappingDao mapping) throws Exception {
        HostDaoStackImpl hosts = new HostDaoStackImpl();
        hosts.init();
        inject(hosts, "asyncTaskHostMappingDao", mapping);
        StorageDao storage = mock(StorageDao.class);
        when(storage.getAsyncTaskRuleEngineInput(anyString())).thenReturn(Future.succeededFuture(new RuleEngineInput()));
        when(storage.getBreakpoints(anyString())).thenReturn(Future.succeededFuture());
        when(storage.getCheckpoint(anyString())).thenReturn(Future.succeededFuture());
        inject(hosts, "storageDao", storage);
        NativeDeliveryDao delivery = mock(NativeDeliveryDao.class);
        when(delivery.claim(any())).thenReturn(Future.succeededFuture(true));
        inject(hosts, "nativeDeliveryDao", delivery);
        return hosts;
    }

    @Test
    public void orchestrationStoresTaggedTaskBeforeCreatingHostMapping() throws Exception {
        OrchestrateService service = new OrchestrateService();
        AsyncTaskDao dao = mock(AsyncTaskDao.class);
        when(dao.insert(any())).thenReturn(Future.succeededFuture());
        HostsDao hosts = mock(HostsDao.class);
        when(hosts.getMachine(any(), eq(false))).thenReturn(Future.succeededFuture("a"));
        inject(service, "asyncTaskDao", dao);
        inject(service, "hostsDao", hosts);
        assertTrue(service.orchestrateService("command", "request", false, Collections.emptyList(), "A").succeeded());
        ArgumentCaptor<AsyncTask> captured = ArgumentCaptor.forClass(AsyncTask.class);
        org.mockito.InOrder order = inOrder(dao, hosts);
        order.verify(dao).insert(captured.capture());
        order.verify(hosts).getMachine(any(), eq(false));
        assertEquals("A", captured.getValue().getClusterId());
    }

    @Test
    public void taggedTasksSelectAndRetrieveOnlyMatchingDevicesAndHostsCanRepoll() throws Exception {
        AsyncTaskHostMappingDaoHashMapImpl mapping = new AsyncTaskHostMappingDaoHashMapImpl();
        mapping.init();
        HostDaoStackImpl hosts = hosts(mapping);
        hosts.putMachineForComputation("b", "B");
        hosts.putMachineForComputation("global");
        AsyncTask task = task("tagged", "A");
        assertEquals("No Machine available", hosts.getMachine(task, false).result());
        hosts.putMachineForComputation("a", "A");
        assertEquals("a", hosts.getMachine(task, false).result());
        assertNull(hosts.putMachineForComputation("a", "B").result());
        assertNull(hosts.putMachineForComputation("a").result());
        assertEquals("A", hosts.putMachineForComputation("a", "A").result().getClusterId());
        mapping.removeMapping("a", task.getUuid());
        hosts.putMachineForComputation("a", "A");
        assertEquals("a", hosts.getMachine(task("again", "A"), false).result());
    }

    @Test
    public void untaggedTasksAcceptEveryClusterAndStaleHostsAreRemoved() throws Exception {
        for (String cluster : new String[]{null, "A", "B"}) {
            AsyncTaskHostMappingDaoHashMapImpl mapping = new AsyncTaskHostMappingDaoHashMapImpl();
            mapping.init();
            HostDaoStackImpl hosts = hosts(mapping);
            hosts.putMachineForComputation("worker", cluster);
            AsyncTask task = task("global", null);
            assertEquals("worker", hosts.getMachine(task, false).result());
            assertEquals(cluster, task.getAssignedCluster());
            assertNotNull(hosts.putMachineForComputation("worker", cluster).result());
        }
        AsyncTaskHostMappingDaoHashMapImpl mapping = new AsyncTaskHostMappingDaoHashMapImpl();
        mapping.init();
        HostDaoStackImpl hosts = hosts(mapping);
        hosts.putMachineForComputation("stale", "A");
        Field field = HostDaoStackImpl.class.getDeclaredField("hostLastSeen");
        field.setAccessible(true);
        ((Map<String, Long>) field.get(hosts)).put("stale", 0L);
        assertEquals("No Machine available", hosts.getMachine(task("tagged", "A"), false).result());
        hosts.putMachineForComputation("stale", "A");
        assertEquals("stale", hosts.getMachine(task("new", "A"), false).result());
    }

    @Test
    public void sqlTasksAssignmentsAndRetryRetainCluster() throws Exception {
        QueryExecutor query = new QueryExecutor();
        query.init(null, QueryExecutor.DB_TYPE.IN_MEM, null);
        AsyncTaskDaoSqlImpl dao = new AsyncTaskDaoSqlImpl();
        inject(dao, "queryExecutor", query);
        AsyncTaskMappingSqlDbImpl mapping = new AsyncTaskMappingSqlDbImpl();
        inject(mapping, "queryExecutor", query);
        inject(mapping, "asyncTaskDao", dao);
        HostDaoStackImpl hosts = hosts(mapping);
        AsyncTask task = task("sql-task", "A");
        assertTrue(dao.insert(task).succeeded());
        hosts.putMachineForComputation("old-a", "A");
        assertEquals("old-a", hosts.getMachine(task, false).result());
        assertEquals("A", dao.getAsyncTask(task.getUuid()).result().getClusterId());
        assertEquals("A", mapping.getMapping("old-a").result().getAssignedCluster());
        hosts.putMachineForComputation("new-a", "A");
        hosts.putMachineForComputation("b", "B");
        hosts.putMachineForComputation("global");
        OrchestratorTaskStatusService status = new OrchestratorTaskStatusService();
        inject(status, "asyncTaskDao", dao);
        inject(status, "asyncTaskHostMappingDao", mapping);
        inject(status, "hostsDao", hosts);
        HeartBeatDao heartbeat = mock(HeartBeatDao.class);
        when(heartbeat.getLastHeartBeat(anyString(), anyString())).thenReturn(Future.succeededFuture());
        inject(status, "heartBeatDao", heartbeat);
        AsyncTask retried = status.getAsyncTaskStatus(task.getUuid()).result();
        assertEquals("new-a", retried.getHostAssigned());
        assertEquals("A", retried.getClusterId());
        assertEquals("A", mapping.getMapping("new-a").result().getClusterId());
        assertTrue(status.getAsyncTaskStatus(task.getUuid(), "B").cause() instanceof SecurityException);
        assertTrue(status.getAsyncTaskStatus(task.getUuid(), null).cause() instanceof SecurityException);
    }

    @Test
    public void checkpointResumeLoadsPersistedTaskInsteadOfDroppingTag() throws Exception {
        OrchestratorCheckpointResumeService service = new OrchestratorCheckpointResumeService();
        AsyncTaskDao dao = mock(AsyncTaskDao.class);
        AsyncTask stored = task("resume", "A");
        when(dao.getAsyncTask("resume")).thenReturn(Future.succeededFuture(stored));
        StorageDao storage = mock(StorageDao.class);
        when(storage.storeBreakpoints(eq("resume"), any())).thenReturn(Future.succeededFuture());
        HostsDao hosts = mock(HostsDao.class);
        when(hosts.getMachine(any(), eq(true))).thenReturn(Future.succeededFuture("a"));
        inject(service, "asyncTaskDao", dao);
        inject(service, "storageDao", storage);
        inject(service, "hostsDao", hosts);
        assertTrue(service.resumeCheckpoint("resume", new CheckpointResumePayload()).succeeded());
        ArgumentCaptor<AsyncTask> captured = ArgumentCaptor.forClass(AsyncTask.class);
        verify(hosts).getMachine(captured.capture(), eq(true));
        assertEquals("A", captured.getValue().getClusterId());
    }

    @Test
    public void completionRejectsWrongHostAndClusterBeforeWritingResults() throws Exception {
        AsyncTaskHostMappingDaoHashMapImpl mapping = new AsyncTaskHostMappingDaoHashMapImpl();
        mapping.init();
        AsyncTask task = task("complete", "A");
        task.setAssignedCluster("A");
        mapping.createMapping(task, "a", false);
        TaskCompleteService service = new TaskCompleteService();
        inject(service, "asyncTaskHostMappingDao", mapping);
        StorageDao storage = mock(StorageDao.class);
        when(storage.storeAsyncTaskResult(anyString(), anyString())).thenReturn(Future.succeededFuture());
        inject(service, "storageDao", storage);
        AsyncTaskDao dao = mock(AsyncTaskDao.class);
        when(dao.update(anyString(), anyMap())).thenReturn(Future.succeededFuture());
        inject(service, "asyncTaskDao", dao);
        assertTrue(service.completeTask("b", "complete", Collections.emptyMap(), "A").failed());
        assertTrue(service.completeTask("a", "complete", Collections.emptyMap(), "B").failed());
        assertTrue(service.completeTask("a", "complete", Collections.emptyMap()).failed());
        verifyZeroInteractions(storage);
        assertTrue(service.completeTask("a", "complete", Collections.emptyMap(), "A").succeeded());
        assertNull(mapping.getMapping("a").result());
    }

    @Test
    public void registrationForwardingPreservesCluster() throws Exception {
        HostDaoStackImpl hosts = new HostDaoStackImpl();
        hosts.init();
        Field field = HostDaoStackImpl.class.getDeclaredField("hostStack");
        field.setAccessible(true);
        Stack<String> stack = (Stack<String>) field.get(hosts);
        for (int i = 0; i < 10000; i++) stack.push("worker-" + i);
        OrchestratorApiCaller caller = mock(OrchestratorApiCaller.class);
        when(caller.callOpenPingApiWithRetry("forward", 3, "A")).thenReturn(Future.succeededFuture());
        inject(hosts, "orchestratorApiCaller", caller);
        assertTrue(hosts.putMachineForComputation("forward", "A").succeeded());
        verify(caller).callOpenPingApiWithRetry("forward", 3, "A");
    }
}

package in.ramanujan.orchestrator.service;

import in.ramanujan.db.layer.schema.NativeAffinityOwner;
import in.ramanujan.db.layer.utils.QueryExecutor;
import in.ramanujan.orchestrator.base.pojo.AsyncTask;
import in.ramanujan.orchestrator.data.dao.*;
import in.ramanujan.orchestrator.data.impl.asyncTaskDaoImpl.AsyncTaskDaoSqlImpl;
import in.ramanujan.orchestrator.data.impl.asyncTaskHostMappingDaoImpl.AsyncTaskMappingSqlDbImpl;
import in.ramanujan.orchestrator.data.impl.hostDaoImpl.HostDaoStackImpl;
import io.vertx.core.Future;
import io.vertx.core.Vertx;
import org.junit.Test;

import java.lang.reflect.Field;
import java.util.*;
import java.util.concurrent.*;

import static org.junit.Assert.*;
import static org.mockito.Mockito.*;

public class NativeLlmRoutingTest {
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

    private static <T> T await(Future<T> future) throws Exception {
        CompletableFuture<T> result = new CompletableFuture<>();
        future.setHandler(done -> {
            if (done.succeeded()) result.complete(done.result()); else result.completeExceptionally(done.cause());
        });
        return result.get(5, TimeUnit.SECONDS);
    }

    private static final class Fixture implements AutoCloseable {
        final Vertx vertx = Vertx.vertx();
        final QueryExecutor query = new QueryExecutor();
        final AsyncTaskDaoSqlImpl tasks = new AsyncTaskDaoSqlImpl();
        final AsyncTaskMappingSqlDbImpl mappings = new AsyncTaskMappingSqlDbImpl();
        final NativeAffinityDao affinities = new NativeAffinityDao();
        final NativeDeliveryDao deliveries = new NativeDeliveryDao();
        final HostDaoStackImpl hosts = new HostDaoStackImpl();
        final NativeLlmService llm;
        final TaskCompleteService complete = new TaskCompleteService();

        Fixture() throws Exception {
            query.init(null, QueryExecutor.DB_TYPE.IN_MEM, null);
            inject(tasks, "queryExecutor", query);
            inject(affinities, "queryExecutor", query);
            inject(affinities, "asyncTaskDao", tasks);
            inject(deliveries, "queryExecutor", query);
            inject(mappings, "queryExecutor", query);
            inject(mappings, "asyncTaskDao", tasks);
            inject(mappings, "nativeAffinityDao", affinities);
            hosts.init();
            inject(hosts, "asyncTaskHostMappingDao", mappings);
            inject(hosts, "nativeAffinityDao", affinities);
            inject(hosts, "nativeDeliveryDao", deliveries);
            llm = service();
            inject(complete, "asyncTaskHostMappingDao", mappings);
            inject(complete, "asyncTaskDao", tasks);
            inject(complete, "nativeLlmService", llm);
        }

        NativeLlmService service() throws Exception {
            NativeLlmService llm = new NativeLlmService();
            inject(llm, "asyncTaskDao", tasks);
            inject(llm, "hostMappingDao", mappings);
            inject(llm, "hostsDao", hosts);
            inject(llm, "affinityDao", affinities);
            return llm;
        }

        AsyncTask poll(String host, String cluster) throws Exception {
            long deadline = System.currentTimeMillis() + 4000;
            while (System.currentTimeMillis() < deadline) {
                AsyncTask task = await(hosts.putMachineForComputation(host, cluster, 8));
                if (task != null) return task;
                Thread.sleep(20);
            }
            throw new AssertionError("No native task for " + host + "/" + cluster);
        }

        void finish(AsyncTask task, String host, String cluster, Map<String, Object> data) throws Exception {
            await(complete.completeTask(host, task.getUuid(), data, cluster));
        }

        @Override public void close() throws Exception {
            CountDownLatch closed = new CountDownLatch(1);
            vertx.close(done -> closed.countDown());
            assertTrue(closed.await(5, TimeUnit.SECONDS));
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    public void taggedNativeWorkPersistsAndDeliversOnceToOnlyMatchingRoom() throws Exception {
        try (Fixture f = new Fixture()) {
            f.hosts.putMachineForComputation("b", "B");
            f.hosts.putMachineForComputation("global");
            f.hosts.putMachineForComputation("a", "A");
            Future<Map<String, Object>> result = f.llm.execute(map("clusterId", "A", "affinity", "shard",
                    "session", "chat", "graph", map("weights", "approved.bin"),
                    "files", Arrays.asList("approved.bin"), "timeout", 5), "step", f.vertx);
            AsyncTask task = f.poll("a", "A");
            assertEquals("A", task.getClusterId());
            assertEquals("ASSIGNED", await(f.tasks.getAsyncTask(task.getUuid())).getNativeState());
            assertEquals(task.getLlm(), await(f.tasks.getAsyncTask(task.getUuid())).getLlm());
            assertNull(await(f.hosts.putMachineForComputation("b", "B")));
            assertNull(await(f.hosts.putMachineForComputation("global")));
            assertNull("parallel poll must not execute same stateful step twice",
                    await(f.hosts.putMachineForComputation("a", "A")));
            assertTrue(f.complete.completeTask("b", task.getUuid(), map("token", 1), "A").failed());
            assertTrue(f.complete.completeTask("a", task.getUuid(), map("token", 1), "B").failed());
            f.finish(task, "a", "A", map("token", 42));
            assertEquals(42, ((Map<String, Object>) await(result).get("data")).get("token"));
            AsyncTask stored = await(f.tasks.getAsyncTask(task.getUuid()));
            assertEquals("COMPLETED", stored.getNativeState());
            assertNotNull(stored.getNativeResult());
            assertNull(await(f.mappings.getMapping("a")));
            assertTrue(f.llm.execute(map("clusterId", "A", "affinity", "shard",
                    "session", "chat", "timeout", 5), "step", f.vertx).failed());
        }
    }

    @Test
    public void sessionOwnershipSurvivesServiceRecreationAndOtherRoomUsesDifferentSession() throws Exception {
        try (Fixture f = new Fixture()) {
            f.hosts.putMachineForComputation("a", "A");
            Future<Map<String, Object>> first = f.llm.execute(map("clusterId", "A", "affinity", "same",
                    "session", "chat", "files", Arrays.asList("room-a.bin"), "timeout", 5), "step", f.vertx);
            AsyncTask original = f.poll("a", "A");
            f.finish(original, "a", "A", map("token", 1));
            await(first);
            f.hosts.putMachineForComputation("a", "A");
            f.hosts.putMachineForComputation("other-a", "A");
            NativeLlmService recreated = f.service();
            Future<Map<String, Object>> next = recreated.execute(map("clusterId", "A", "affinity", "same",
                    "session", "chat", "pos", 1, "timeout", 5), "step", f.vertx);
            AsyncTask resumed = f.poll("a", "A");
            assertEquals(original.getLlm().get("session"), resumed.getLlm().get("session"));
            assertEquals(Collections.singletonList("room-a.bin"), resumed.getNativeFiles());
            assertNull(await(f.hosts.putMachineForComputation("other-a", "A")));
            f.finish(resumed, "a", "A", map("token", 2));
            await(next);
            Future<Map<String, Object>> other = recreated.execute(map("clusterId", "B", "affinity", "same",
                    "session", "chat", "timeout", 5), "step", f.vertx);
            AsyncTask roomB = f.poll("b", "B");
            assertNotEquals(original.getLlm().get("session"), roomB.getLlm().get("session"));
            assertTrue(roomB.getNativeFiles().isEmpty());
            f.finish(roomB, "b", "B", map("token", 3));
            await(other);
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    public void chainGroupsOnlyDurablyOwnedSameRoomStagesAndCloseReleasesCapacity() throws Exception {
        try (Fixture f = new Fixture()) {
            f.hosts.putMachineForComputation("a", "A");
            for (String shard : Arrays.asList("first", "second")) {
                Future<Map<String, Object>> stage = f.llm.execute(map("clusterId", "A", "affinity", shard,
                        "session", shard, "files", Arrays.asList(shard + ".bin"), "timeout", 5), "step", f.vertx);
                AsyncTask task = f.poll("a", "A");
                f.finish(task, "a", "A", map("output", "hidden"));
                await(stage);
                f.hosts.putMachineForComputation("a", "A");
            }
            Future<Map<String, Object>> chain = f.llm.execute(map("clusterId", "A", "stages", Arrays.asList(
                    map("affinity", "first", "session", "first"),
                    map("affinity", "second", "session", "second")), "pos", 1, "timeout", 5), "chain", f.vertx);
            AsyncTask task = f.poll("a", "A");
            assertEquals(2, ((List<?>) task.getLlm().get("stages")).size());
            assertEquals(new HashSet<>(Arrays.asList("first.bin", "second.bin")), new HashSet<>(task.getNativeFiles()));
            f.finish(task, "a", "A", map("token", 9, "infos", Arrays.asList("first", "second")));
            Map<String, Object> response = await(chain);
            assertEquals(1, response.get("tasks"));
            assertEquals(9, response.get("token"));
            Future<Map<String, Object>> close = f.llm.execute(map("clusterId", "A", "affinity", "first",
                    "session", "first", "timeout", 5), "close", f.vertx);
            AsyncTask closing = f.poll("a", "A");
            f.finish(closing, "a", "A", map("closed", true));
            await(close);
            NativeAffinityOwner owner = await(f.affinities.get(NativeLlmService.binding("A", "first", "first")));
            assertEquals("CLOSED", owner.getState());
        }
    }

    @Test
    public void lostAssignmentTimesOutFailsSessionAndStatusCannotSilentlyRedispatch() throws Exception {
        try (Fixture f = new Fixture()) {
            f.hosts.putMachineForComputation("a", "A");
            Future<Map<String, Object>> result = f.llm.execute(map("clusterId", "A", "affinity", "shard",
                    "session", "lost", "timeout", 1), "step", f.vertx);
            AsyncTask task = f.poll("a", "A");
            await(f.mappings.deleteTask(task.getUuid()));
            f.hosts.putMachineForComputation("other", "A");
            OrchestratorTaskStatusService status = new OrchestratorTaskStatusService();
            inject(status, "asyncTaskDao", f.tasks);
            inject(status, "asyncTaskHostMappingDao", f.mappings);
            inject(status, "hostsDao", f.hosts);
            assertNull(await(status.getAsyncTaskStatus(task.getUuid())).getHostAssigned());
            assertNull(await(f.hosts.putMachineForComputation("other", "A")));
            assertNotNull(await(result).get("error"));
            assertEquals("FAILED", await(f.tasks.getAsyncTask(task.getUuid())).getNativeState());
            assertTrue(f.llm.execute(map("clusterId", "A", "affinity", "shard", "session", "lost",
                    "timeout", 1), "step", f.vertx).failed());
            assertTrue(f.complete.completeTask("a", task.getUuid(), map("token", 1), "A").failed());
        }
    }

    @Test
    public void untaggedNativeWorkCanUseClusteredDeviceAndNewBindingsSpreadAcrossDevices() throws Exception {
        try (Fixture f = new Fixture()) {
            f.hosts.putMachineForComputation("a", "A", 1);
            Future<Map<String, Object>> global = f.llm.execute(map("affinity", "global", "session", "g",
                    "timeout", 5), "step", f.vertx);
            AsyncTask untagged = f.poll("a", "A");
            assertNull(untagged.getClusterId());
            assertEquals("A", untagged.getAssignedCluster());
            f.finish(untagged, "a", "A", map("token", 1));
            await(global);
            f.hosts.putMachineForComputation("a", "A", 1);
            f.hosts.putMachineForComputation("b", "A", 1);
            Future<Map<String, Object>> next = f.llm.execute(map("clusterId", "A", "affinity", "next",
                    "session", "n", "timeout", 5), "step", f.vertx);
            AsyncTask task = f.poll("b", "A");
            assertEquals("b", task.getHostAssigned());
            f.finish(task, "b", "A", map("token", 2));
            await(next);
        }
    }
}

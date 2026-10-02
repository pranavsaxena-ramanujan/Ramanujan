package in.ramanujan.orchestrator.data.impl.asyncTaskHostMappingDaoImpl;

import in.ramanujan.orchestrator.base.pojo.AsyncTask;
import in.ramanujan.orchestrator.data.dao.AsyncTaskHostMappingDao;
import io.vertx.core.Future;

import java.util.HashMap;
import java.util.Map;

public class AsyncTaskHostMappingDaoHashMapImpl implements AsyncTaskHostMappingDao {

    Map<String, AsyncTask> mapping;

    public void init() {
        mapping = new HashMap<>();
    }

    @Override
    public synchronized Future<Void> createMapping(AsyncTask asyncTask, String hostMachineId, Boolean resumeComputation) {
        /*
        * has to be upserted
        * */
        if (mapping.containsKey(hostMachineId)) return Future.failedFuture("Host is already assigned");
        asyncTask.setHostAssigned(hostMachineId);
        asyncTask.setAssignedNonce(java.util.UUID.randomUUID().toString());
        if (Boolean.TRUE.equals(resumeComputation) && asyncTask.getCheckpoint() == null) {
            asyncTask.setCheckpoint(new in.ramanujan.pojo.checkpoint.Checkpoint());
        }
        mapping.put(hostMachineId, asyncTask);
        return Future.succeededFuture();
    }

    @Override
    public synchronized Future<AsyncTask> getMapping(String hostMachineId) {
        return  Future.succeededFuture(mapping.get(hostMachineId));
    }

    @Override
    public synchronized Future<Void> deleteTask(String asyncTaskId) {
        mapping.entrySet().removeIf(entry -> asyncTaskId.equals(entry.getValue().getUuid()));
        return Future.succeededFuture();
    }

    @Override
    public synchronized Future<String> getHostForTask(String asyncTaskId) {
        for (Map.Entry<String, AsyncTask> entry : mapping.entrySet()) {
            if (asyncTaskId.equals(entry.getValue().getUuid())) return Future.succeededFuture(entry.getKey());
        }
        return Future.succeededFuture();
    }

    @Override
    public synchronized Future<Void> update(String hostMachineId, AsyncTask asyncTask) {
        mapping.put(hostMachineId, asyncTask);
        return Future.succeededFuture();
    }

    @Override
    public synchronized Future<Void> removeMapping(String hostMachineId, String asyncTaskId) {
        AsyncTask task = mapping.get(hostMachineId);
        if (task != null && asyncTaskId.equals(task.getUuid())) mapping.remove(hostMachineId);
        return Future.succeededFuture();
    }
}

package in.ramanujan.orchestrator.data.dao;

import in.ramanujan.orchestrator.base.pojo.AsyncTask;
import io.vertx.core.Future;
import org.springframework.stereotype.Component;

@Component
public interface HostsDao {
    public Future<String> getMachine(AsyncTask asyncTask, Boolean resumeComputation);
    public Future<AsyncTask> putMachineForComputation(String hostId);

    default Future<AsyncTask> putMachineForComputation(String hostId, String clusterId) {
        if (clusterId != null) return Future.failedFuture("Cluster routing is not supported by this host DAO");
        return putMachineForComputation(hostId);
    }

    default Future<AsyncTask> putMachineForComputation(String hostId, String clusterId, int affinityLimit) {
        return putMachineForComputation(hostId, clusterId);
    }
}

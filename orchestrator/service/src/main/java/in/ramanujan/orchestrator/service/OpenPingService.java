package in.ramanujan.orchestrator.service;

import in.ramanujan.orchestrator.base.pojo.AsyncTask;
import in.ramanujan.orchestrator.data.dao.HostsDao;
import io.vertx.core.Future;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;



@Component
public class OpenPingService {

    @Autowired
    private HostsDao hostsDao;

    public Future<AsyncTask> storePing(String hostId) {
        return storePing(hostId, null);
    }

    public Future<AsyncTask> storePing(String hostId, String clusterId) {
        return storePing(hostId, clusterId, Integer.MAX_VALUE);
    }

    public Future<AsyncTask> storePing(String hostId, String clusterId, int affinityLimit) {
        if (hostId == null || hostId.isEmpty()) return Future.failedFuture("uuid is required");
        Future<AsyncTask> future = Future.future();
        hostsDao.putMachineForComputation(hostId, clusterId, affinityLimit).setHandler(handler -> {
           if(handler.succeeded()) {
               future.complete(handler.result());
           } else {
               future.fail(handler.cause());
           }
        });
        return future;
    }
}

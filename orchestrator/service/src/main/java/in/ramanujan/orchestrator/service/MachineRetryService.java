package in.ramanujan.orchestrator.service;

import in.ramanujan.base.pojo.MachineAssignmentTask;
import in.ramanujan.data.MachineRetryCallback;
import in.ramanujan.orchestrator.base.enums.Status;
import in.ramanujan.orchestrator.base.pojo.AsyncTask;
import in.ramanujan.orchestrator.data.dao.AsyncTaskDao;
import in.ramanujan.orchestrator.data.dao.HostsDao;
import io.vertx.core.Future;
import io.vertx.core.Vertx;
import io.vertx.core.logging.Logger;
import io.vertx.core.logging.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class MachineRetryService implements MachineRetryCallback {

    private Logger logger = LoggerFactory.getLogger(MachineRetryService.class);

    @Autowired
    private HostsDao hostsDao;

    @Autowired
    private AsyncTaskDao asyncTaskDao;

    @Override
    public Future<Boolean> retryMachineAssignment(MachineAssignmentTask task, Vertx vertx) {
        Future<Boolean> future = Future.future();
        
        try {
            String asyncId = task.getAsyncId();
            logger.info("Retrying machine assignment for asyncId: " + asyncId);
            
            // Convert MachineAssignmentTask to AsyncTask
            AsyncTask asyncTask = new AsyncTask();
            asyncTask.setUuid(asyncId);
            asyncTask.setStatus(task.getStatus());
            asyncTask.setHostAssigned(task.getHostAssigned());
            // Set other fields as needed
            
            // Try to get a machine
            hostsDao.getMachine(asyncTask, false).setHandler(hostMachineGetHandler -> {
                if (hostMachineGetHandler.succeeded()) {
                    String machine = hostMachineGetHandler.result();
                    logger.info(asyncId + " got machine " + machine);
                    asyncTask.setHostAssigned(machine);
                    
                    future.complete(true);
                } else {
                    logger.error(asyncId + " couldn't find machine", hostMachineGetHandler.cause());
                    future.complete(false);
                }
            });
        } catch (Exception e) {
            logger.error("Error retrying machine assignment", e);
            future.complete(false);
        }
        
        return future;
    }
}

package in.ramanujan.orchestrator.service;

import in.ramanujan.base.pojo.MachineAssignmentTask;
import in.ramanujan.data.MachineRetryCallback;
import io.vertx.core.Future;
import io.vertx.core.Vertx;
import io.vertx.core.logging.Logger;
import io.vertx.core.logging.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/**
 * Implementation of the MachineRetryCallback that will be provided to the MiddlewareClient
 */
@Component
public class MachineRetryManager {
    
    private Logger logger = LoggerFactory.getLogger(MachineRetryManager.class);
    
    @Autowired
    private MachineRetryService machineRetryService;
    
    /**
     * Provides an implementation of the MachineRetryCallback that delegates to the MachineRetryService
     * @return An implementation of MachineRetryCallback
     */
    public MachineRetryCallback getCallback() {
        return new MachineRetryCallback() {
            @Override
            public Future<Boolean> retryMachineAssignment(MachineAssignmentTask task, Vertx vertx) {
                logger.info("Delegating machine retry for task " + task.getAsyncId());
                return machineRetryService.retryMachineAssignment(task, vertx);
            }
        };
    }
}

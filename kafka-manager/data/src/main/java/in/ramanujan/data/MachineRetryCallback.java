package in.ramanujan.data;

import in.ramanujan.base.pojo.MachineAssignmentTask;
import io.vertx.core.Future;
import io.vertx.core.Vertx;

/**
 * Interface for retrying machine assignment for tasks.
 * This will be implemented by the orchestrator service.
 */
public interface MachineRetryCallback {
    /**
     * Tries to assign a machine to the task.
     * @param task The task that needs a machine assignment.
     * @param vertx The Vertx instance.
     * @return A Future that completes with true when the machine is assigned, false if not assigned.
     */
    Future<Boolean> retryMachineAssignment(MachineAssignmentTask task, Vertx vertx);
}

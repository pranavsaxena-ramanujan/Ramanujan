package in.ramanujan.base.pojo;

import java.util.List;

/**
 * Event to be sent to the run-dag topic when a machine cannot be assigned
 */
public class MachineAssignmentQueueEvent {
    private MachineAssignmentTask task;

    public MachineAssignmentQueueEvent() {
    }

    public MachineAssignmentQueueEvent(MachineAssignmentTask task) {
        this.task = task;
    }

    public MachineAssignmentTask getTask() {
        return task;
    }

    public void setTask(MachineAssignmentTask task) {
        this.task = task;
    }
}

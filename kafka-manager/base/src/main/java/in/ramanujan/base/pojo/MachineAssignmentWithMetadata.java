package in.ramanujan.base.pojo;

/**
 * Class for machine assignment events with metadata for pubsub
 */
public class MachineAssignmentWithMetadata {
    private MachineAssignmentTask task;
    private Object metadata;

    public MachineAssignmentWithMetadata() {
    }

    public MachineAssignmentWithMetadata(MachineAssignmentTask task, Object metadata) {
        this.task = task;
        this.metadata = metadata;
    }

    public MachineAssignmentTask getTask() {
        return task;
    }

    public void setTask(MachineAssignmentTask task) {
        this.task = task;
    }

    public Object getMetadata() {
        return metadata;
    }

    public void setMetadata(Object metadata) {
        this.metadata = metadata;
    }
}

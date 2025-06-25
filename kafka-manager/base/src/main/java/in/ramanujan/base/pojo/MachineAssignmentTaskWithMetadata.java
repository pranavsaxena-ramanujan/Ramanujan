package in.ramanujan.base.pojo;

/**
 * Wrapper class for machine assignment task and pubsub metadata
 */
public class MachineAssignmentTaskWithMetadata {
    private MachineAssignmentTask task;
    private String ackId;
    
    public MachineAssignmentTaskWithMetadata() {
    }
    
    public MachineAssignmentTaskWithMetadata(MachineAssignmentTask task, String ackId) {
        this.task = task;
        this.ackId = ackId;
    }
    
    public MachineAssignmentTask getTask() {
        return task;
    }
    
    public void setTask(MachineAssignmentTask task) {
        this.task = task;
    }
    
    public String getAckId() {
        return ackId;
    }
    
    public void setAckId(String ackId) {
        this.ackId = ackId;
    }
}

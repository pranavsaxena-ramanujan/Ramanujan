package in.ramanujan.base.pojo;

import java.util.List;

/**
 * Represents a task that needs machine assignment.
 * Similar to AsyncTask but designed for cross-module compatibility.
 */
public class MachineAssignmentTask {
    private String asyncId;
    private String status;
    private String hostAssigned;
    private String commandId;
    private Boolean debug;
    private List<Integer> debugLines;

    public MachineAssignmentTask() {
    }

    public MachineAssignmentTask(String asyncId, String status, String hostAssigned, String commandId, Boolean debug, List<Integer> debugLines) {
        this.asyncId = asyncId;
        this.status = status;
        this.hostAssigned = hostAssigned;
        this.commandId = commandId;
        this.debug = debug;
        this.debugLines = debugLines;
    }

    public String getAsyncId() {
        return asyncId;
    }

    public void setAsyncId(String asyncId) {
        this.asyncId = asyncId;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getHostAssigned() {
        return hostAssigned;
    }

    public void setHostAssigned(String hostAssigned) {
        this.hostAssigned = hostAssigned;
    }

    public String getCommandId() {
        return commandId;
    }

    public void setCommandId(String commandId) {
        this.commandId = commandId;
    }

    public Boolean getDebug() {
        return debug;
    }

    public void setDebug(Boolean debug) {
        this.debug = debug;
    }

    public List<Integer> getDebugLines() {
        return debugLines;
    }

    public void setDebugLines(List<Integer> debugLines) {
        this.debugLines = debugLines;
    }
}

package in.ramanujan.middleware.base.pojo.asyncTask;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import in.ramanujan.db.layer.schema.AsyncTaskMiddleware;
import lombok.Data;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class AsyncTask {
    private String taskId;
    private String clusterId;
    private TaskStatus taskStatus;
    private Object result;

    public static enum TaskStatus {
        PENDING,
        FAILED,
        SUCCESS,
        CHECKPOINT;
    }

    public AsyncTaskMiddleware getAsyncTaskMiddleware() {
        AsyncTaskMiddleware asyncTaskMiddleware = new AsyncTaskMiddleware();
        asyncTaskMiddleware.setTaskId(taskId);
        asyncTaskMiddleware.setClusterId(clusterId);
        if (taskStatus != null) asyncTaskMiddleware.setTaskStatus(taskStatus.name());
        if(result != null) {
            asyncTaskMiddleware.setResult(result.toString());
        }
        return asyncTaskMiddleware;
    }

    public AsyncTask() {

    }

    public AsyncTask(AsyncTaskMiddleware asyncTaskMiddleware) {
        this.setResult(asyncTaskMiddleware.getResult());
        this.setTaskId(asyncTaskMiddleware.getTaskId());
        this.setClusterId(asyncTaskMiddleware.getClusterId());
        this.setTaskStatus(TaskStatus.valueOf(asyncTaskMiddleware.getTaskStatus()));
    }
}

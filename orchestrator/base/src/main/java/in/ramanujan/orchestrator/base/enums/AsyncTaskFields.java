package in.ramanujan.orchestrator.base.enums;

import in.ramanujan.orchestrator.base.pojo.AsyncTask;
import in.ramanujan.pojo.checkpoint.Checkpoint;

public enum AsyncTaskFields {
    uuid("uuid", (task, object) ->{task.setUuid((String) object);}),
    status("status", ((task, update) -> {task.setStatus((String) update);})),
    nativeState("nativeState", (task, value) -> task.setNativeState((String) value)),
    nativeResult("nativeResult", (task, value) -> task.setNativeResult((java.util.Map<String, Object>) value)),
    binaryArrayFiles("binaryArrayFiles", (task, value) -> task.setBinaryArrayFiles((java.util.Map<String, String>) value)),
//    hostAssigned("hostAssigned", ((task, update) -> {task.setHostAssigned((String) update);})),
//    data("data", ((task, update) -> {task.setData(update);})),
    checkpoint("checkpoint", ((task, update) -> {task.setCheckpoint((Checkpoint) update);}));

    private String fieldName;
    private AsyncTaskFieldsUpdator asyncTaskFieldsUpdator;

    AsyncTaskFields(String fieldName, AsyncTaskFieldsUpdator asyncTaskFieldsUpdator) {
        this.fieldName = fieldName;
        this.asyncTaskFieldsUpdator = asyncTaskFieldsUpdator;
    }

    public void triggerUpdate(AsyncTask asyncTask, Object objectUpdated) {
        asyncTaskFieldsUpdator.updateAsyncTaskFields(asyncTask, objectUpdated);
    }

    public String getFieldName() {
        return fieldName;
    }

    public void setFieldName(String fieldName) {
        this.fieldName = fieldName;
    }

    public static AsyncTaskFields getAsyncTaskFields(String key) {
        for(AsyncTaskFields asyncTaskFields : values()) {
            if(asyncTaskFields.getFieldName().equalsIgnoreCase(key)) {
                return asyncTaskFields;
            }
        }
        return null;
    }


    private static interface AsyncTaskFieldsUpdator {
        public void updateAsyncTaskFields(AsyncTask task, Object update);
    }
}

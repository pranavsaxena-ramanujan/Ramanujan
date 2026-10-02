package in.ramanujan.orchestrator.data.impl.asyncTaskDaoImpl;

import in.ramanujan.db.layer.constants.Keys;
import in.ramanujan.db.layer.enums.QueryType;
import in.ramanujan.db.layer.schema.AsyncTaskOrchestrator;
import in.ramanujan.db.layer.utils.QueryExecutor;
import in.ramanujan.orchestrator.base.enums.AsyncTaskFields;
import in.ramanujan.orchestrator.base.pojo.AsyncTask;
import in.ramanujan.orchestrator.data.dao.AsyncTaskDao;
import io.vertx.core.Future;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

@Component
public class AsyncTaskDaoSqlImpl implements AsyncTaskDao {
    private final com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();

    private void writeNativeFields(AsyncTaskOrchestrator row, AsyncTask task) throws Exception {
        row.setNativeState(task.getNativeState());
        row.setNativeDeadline(task.getNativeDeadline());
        if (task.getLlm() != null) row.setLlm(mapper.writeValueAsString(task.getLlm()));
        if (task.getNativeResult() != null) row.setNativeResult(mapper.writeValueAsString(task.getNativeResult()));
        if (task.getNativeBindings() != null) row.setNativeBindings(mapper.writeValueAsString(task.getNativeBindings()));
        if (task.getNativeFiles() != null) row.setNativeFiles(mapper.writeValueAsString(task.getNativeFiles()));
        if (task.getBinaryArrayFiles() != null) row.setBinaryArrayFiles(mapper.writeValueAsString(task.getBinaryArrayFiles()));
    }

    @Autowired
    private QueryExecutor queryExecutor;

    @Override
    public Future<Void> insert(AsyncTask asyncTask) {
        Future<Void> future = Future.future();
        try {
            AsyncTaskOrchestrator asyncTaskOrchestrator = new AsyncTaskOrchestrator();
            asyncTaskOrchestrator.setFirstCommandId(asyncTask.getFirstCommandId());
            asyncTaskOrchestrator.setStatus(asyncTask.getStatus());
            asyncTaskOrchestrator.setUuid(asyncTask.getUuid());
            asyncTaskOrchestrator.setClusterId(asyncTask.getClusterId());
            writeNativeFields(asyncTaskOrchestrator, asyncTask);
            asyncTaskOrchestrator.setDebug(asyncTask.getDebug().toString());
            queryExecutor.execute(asyncTaskOrchestrator, null, QueryType.INSERT).setHandler(handler -> {
                if(handler.succeeded()) {
                    future.complete();
                } else {
                    future.fail(handler.cause());
                }
            });
        } catch (Exception e) {
            future.fail(e);
        }
        return future;
    }

    @Override
    public Future<Void> update(String uuid, Map<String, Object> updateQuery) {
        Future<Void> future = Future.future();
        try {
            AsyncTaskOrchestrator asyncTaskOrchestrator = new AsyncTaskOrchestrator();
            asyncTaskOrchestrator.setUuid(uuid);
            AsyncTask update = new AsyncTask();
            for (Map.Entry<String, Object> entry : updateQuery.entrySet()) {
                AsyncTaskFields field = AsyncTaskFields.getAsyncTaskFields(entry.getKey());
                if (field == null) throw new IllegalArgumentException("Unknown async task field: " + entry.getKey());
                field.triggerUpdate(update, entry.getValue());
            }
            asyncTaskOrchestrator.setStatus(update.getStatus());
            writeNativeFields(asyncTaskOrchestrator, update);
            queryExecutor.execute(asyncTaskOrchestrator, Keys.UUID, QueryType.UPDATE).setHandler(handler -> {
               if(handler.succeeded()) {
                   future.complete();
               } else {
                   future.fail(handler.cause());
               }
            });
        } catch (Exception e) {
            future.fail(e);
        }
        return future;
    }

    public static class DummyAsyncTask extends AsyncTask {
        public DummyAsyncTask() {
            super();
        }
    }
    @Override
    public Future<AsyncTask> getAsyncTask(String uuid) {
        Future<AsyncTask> future = Future.future();
        try {
            AsyncTaskOrchestrator asyncTaskOrchestrator = new AsyncTaskOrchestrator();
            asyncTaskOrchestrator.setUuid(uuid);
            queryExecutor.execute(asyncTaskOrchestrator, Keys.UUID, QueryType.SELECT).setHandler(handler -> {
               if(handler.succeeded()) {
                   AsyncTask asyncTask = new AsyncTask();
                   List<Object> objectList = handler.result();
                   if(objectList != null && objectList.size() > 0) {
                       AsyncTaskOrchestrator resultObj = (AsyncTaskOrchestrator) objectList.get(0);
                       asyncTask.setStatus(resultObj.getStatus());
                       asyncTask.setClusterId(resultObj.getClusterId());
                       asyncTask.setNativeState(resultObj.getNativeState());
                       asyncTask.setNativeDeadline(resultObj.getNativeDeadline());
                       try {
                           if (resultObj.getLlm() != null) asyncTask.setLlm(mapper.readValue(resultObj.getLlm(), Map.class));
                           if (resultObj.getNativeResult() != null) asyncTask.setNativeResult(mapper.readValue(resultObj.getNativeResult(), Map.class));
                           if (resultObj.getNativeBindings() != null) asyncTask.setNativeBindings(mapper.readValue(resultObj.getNativeBindings(), List.class));
                           if (resultObj.getNativeFiles() != null) asyncTask.setNativeFiles(mapper.readValue(resultObj.getNativeFiles(), List.class));
                           if (resultObj.getBinaryArrayFiles() != null) asyncTask.setBinaryArrayFiles(mapper.readValue(resultObj.getBinaryArrayFiles(), Map.class));
                       } catch (Exception ex) {
                           future.fail(ex);
                           return;
                       }
                       asyncTask.setFirstCommandId(resultObj.getFirstCommandId());
                       asyncTask.setDebug(Boolean.valueOf(resultObj.getDebug()));
                       asyncTask.setUuid(uuid);
                   } else {
                       asyncTask = new DummyAsyncTask();
                   }
                   future.complete(asyncTask);
               } else {
                   future.fail(handler.cause());
               }
            });
        } catch (Exception e) {
            future.fail(e);
        }
        return future;
    }
}

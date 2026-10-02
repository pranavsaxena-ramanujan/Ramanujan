package in.ramanujan.middleware.service;

import in.ramanujan.data.db.dao.AsyncTaskDao;
import in.ramanujan.data.db.dao.DagElementDao;
import in.ramanujan.data.db.dao.VariableValueDao;
import in.ramanujan.middleware.base.pojo.asyncTask.AsyncTask;
import in.ramanujan.translation.codeConverter.pojo.VariableAndArrayResult;
import io.vertx.core.Future;
import io.vertx.core.Promise;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class TaskStatusService {

    @Autowired
    private AsyncTaskDao asyncTaskDao;

    @Autowired
    private VariableValueDao variableValueDao;

    @Autowired
    private DagElementDao dagElementDao;

    public Future<AsyncTask> getAsyncTaskStatus(String asyncTaskId) {
        return getAsyncTaskStatus(asyncTaskId, null);
    }

    public Future<AsyncTask> getAsyncTaskStatus(String asyncTaskId, String clusterId) {
        Future<AsyncTask> future = Future.future();
        asyncTaskDao.getAsyncTask(asyncTaskId).setHandler(taskHandler -> {
            if (taskHandler.failed()) {
                future.fail(taskHandler.cause());
                return;
            }
            AsyncTask stored = taskHandler.result();
            if (stored == null || !java.util.Objects.equals(stored.getClusterId(), clusterId)) {
                future.fail(new SecurityException("Unknown task or wrong cluster"));
                return;
            }
        dagElementDao.isAsyncTaskDone(asyncTaskId).setHandler(asyncDoneHandler -> {
           if(asyncDoneHandler.succeeded()) {
               if(asyncDoneHandler.result() == 0 ) {
                   AsyncTask asyncTask = new AsyncTask();
                   asyncTask.setClusterId(stored.getClusterId());
                     asyncTask.setTaskId(asyncTaskId);
                        asyncTask.setTaskStatus(AsyncTask.TaskStatus.SUCCESS);
                   variableValueDao.getAllValuesForAsyncId(asyncTaskId).setHandler(getValueHandler -> {
                          if(getValueHandler.failed()) {
                            future.fail(getValueHandler.cause());
                            return;
                          }
                       VariableAndArrayResult result = getValueHandler.result();
                       asyncTask.setResult(result);
                       future.complete(asyncTask);
                     });
               } else {
                   future.complete(stored);
               }
           } else {
               future.fail(asyncDoneHandler.cause());
               return;
           }
        });
        });
        return  future;
    }
}

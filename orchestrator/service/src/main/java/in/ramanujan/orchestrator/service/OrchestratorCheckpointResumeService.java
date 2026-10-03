package in.ramanujan.orchestrator.service;

import in.ramanujan.orchestrator.base.pojo.AsyncTask;
import in.ramanujan.orchestrator.base.pojo.CheckpointResumePayload;
import in.ramanujan.orchestrator.data.dao.HostsDao;
import in.ramanujan.orchestrator.data.dao.StorageDao;
import io.vertx.core.Future;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class OrchestratorCheckpointResumeService {

    @Autowired
    private StorageDao storageDao;

    @Autowired
    private HostsDao hostsDao;

    @Autowired
    private in.ramanujan.orchestrator.data.dao.AsyncTaskDao asyncTaskDao;

    /**
     * <pre>
     * 1. add resume checkpoint on storage.
     * 2. select machine for processing.
     * </pre>
     */
    public Future<Void> resumeCheckpoint(String asyncId, CheckpointResumePayload checkpointResumePayload) {
        Future<Void> future = Future.future();
        storageDao.storeBreakpoints(asyncId, checkpointResumePayload).setHandler(storeBreakpointHandler -> {
           if(storeBreakpointHandler.succeeded()) {
               asyncTaskDao.getAsyncTask(asyncId).setHandler(taskHandler -> {
                   if (taskHandler.failed()) {
                       future.fail(taskHandler.cause());
                       return;
                   }
                   AsyncTask asyncTask = taskHandler.result();
                   if (asyncTask == null || asyncTask.getUuid() == null) {
                       future.fail("Unknown async task: " + asyncId);
                       return;
                   }
                   hostsDao.getMachine(asyncTask, true).setHandler(getMachineHandler -> {
                  if(getMachineHandler.succeeded()) {
                      future.complete();
                  } else {
                      future.fail(getMachineHandler.cause());
                  }
               });
               });
           } else {
               future.fail(storeBreakpointHandler.cause());
           }
        });
        return future;
    }
}

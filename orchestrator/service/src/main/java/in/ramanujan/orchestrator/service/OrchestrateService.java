package in.ramanujan.orchestrator.service;

import in.ramanujan.orchestrator.base.enums.Status;
import in.ramanujan.orchestrator.base.pojo.AsyncTask;
import in.ramanujan.orchestrator.base.pojo.CheckpointResumePayload;
import in.ramanujan.orchestrator.data.dao.AsyncTaskDao;
import in.ramanujan.orchestrator.data.dao.HostsDao;
import in.ramanujan.orchestrator.data.dao.StorageDao;
import io.vertx.core.Future;
import io.vertx.core.Vertx;
import io.vertx.core.logging.Logger;
import io.vertx.core.logging.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;


/*
* https://docs.google.com/document/d/16UdGECC2vymneZfSsOTL-N_EH3ZqbWLADaBmeI8LTAE/edit
* */

@Component
public class OrchestrateService {

    private Logger logger= LoggerFactory.getLogger(OrchestrateService.class);
    @Autowired
    private HostsDao hostsDao;

    @Autowired
    private AsyncTaskDao asyncTaskDao;

    @Autowired
    private StorageDao storageDao;

    public Future<Void> orchestrateService(String firstCommandId, String orchestratorAsyncId, Boolean debug, List<Integer> debugLines) {
        return orchestrateService(firstCommandId, orchestratorAsyncId, debug, debugLines, null);
    }

    public Future<Void> orchestrateService(String firstCommandId, String orchestratorAsyncId, Boolean debug,
                                          List<Integer> debugLines, String clusterId) {
        Future<Void> future = Future.future();
        AsyncTask asyncTask = new AsyncTask(orchestratorAsyncId, Status.PROCESSING.getKeyName(),
               null, null, null, firstCommandId, null, debug, debugLines);
        asyncTask.setClusterId(clusterId);
        if(debugLines != null && debugLines.size() > 0) {
            CheckpointResumePayload payload = new CheckpointResumePayload();
            payload.setLines(debugLines);
            storageDao.storeBreakpoints(orchestratorAsyncId, payload).setHandler(handler -> {
                if(handler.succeeded()) {
                    assignMachine(orchestratorAsyncId, future, asyncTask);
                } else {
                    future.fail(handler.cause());
                }
            });
        } else {
            assignMachine(orchestratorAsyncId, future, asyncTask);
        }
//        refreshVariables(asyncId, ruleEngineInput, dagElementId).setHandler(handler -> {
//           if(handler.succeeded()) {
//               assignMachine(asyncId, future, asyncTask);
//           } else {
//               future.fail(handler.cause());
//           }
//        });
        return future;
    }

    private void assignMachine(String asyncId, Future<Void> future, AsyncTask asyncTask) {
        // Make the task visible before its host mapping can be retrieved by a polling worker.
        asyncTaskDao.insert(asyncTask).setHandler(insertHandler -> {
            if (insertHandler.failed()) {
                future.fail(insertHandler.cause());
                return;
            }
            hostsDao.getMachine(asyncTask, false).setHandler(hostMachineGetHandler -> {
                if (hostMachineGetHandler.succeeded()) {
                    asyncTask.setHostAssigned(hostMachineGetHandler.result());
                    future.complete();
                } else {
                    future.fail(hostMachineGetHandler.cause());
                }
            });
        });
    }

    private Object setValueAsPerDataType(Object value, String dataType) {
        try {
            if(value.toString().contains(".")) {
                return Double.parseDouble(value.toString());
            }
        } catch (Exception e) {

        }

        try {
            return Integer.parseInt(value.toString());
        } catch (Exception e) {

        }


        return value;
    }
}

package in.ramanujan.orchestrator.service;

import in.ramanujan.base.pojo.MachineAssignmentQueueEvent;
import in.ramanujan.base.pojo.MachineAssignmentTask;
import in.ramanujan.data.QueueingDao;
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

import java.util.HashMap;
import java.util.List;
import java.util.Map;


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
    
    @Autowired
    private QueueingDao queueingDao;

    public Future<Void> orchestrateService(String firstCommandId, String orchestratorAsyncId, Boolean debug, List<Integer> debugLines) {
        Future<Void> future = Future.future();
        AsyncTask asyncTask = new AsyncTask(orchestratorAsyncId, Status.PROCESSING.getKeyName(),
               null, null, null, firstCommandId, null, debug, debugLines);
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
        // First insert the AsyncTask into the database before trying to find a machine
        asyncTaskDao.insert(asyncTask).setHandler(asyncTaskInsertHandler -> {
            if (asyncTaskInsertHandler.succeeded()) {
                logger.info(asyncId + " inserted asyncTask in asyncTaskDataStore before machine assignment");
                
                // Now try to find a machine
                hostsDao.getMachine(asyncTask, false).setHandler(hostMachineGetHandler -> {
                    if (hostMachineGetHandler.succeeded()) {
                        logger.info(asyncId + " got machine " + hostMachineGetHandler.result());
                        asyncTask.setHostAssigned(hostMachineGetHandler.result());
                        future.complete();
                    } else {
                        logger.error(asyncId + " couldn't find machine", hostMachineGetHandler.cause());
                        // Push to PubSub queue to retry machine assignment later
                        try {
                            // Create a MachineAssignmentTask from AsyncTask
                            MachineAssignmentTask task = new MachineAssignmentTask(
                                asyncTask.getUuid(),  // AsyncTask uses uuid, not asyncId
                                asyncTask.getStatus(),
                                asyncTask.getHostAssigned(),
                                asyncTask.getFirstCommandId(),
                                asyncTask.getDebug(),
                                asyncTask.getBreakpoints() // AsyncTask uses breakpoints, not debugLines
                            );
                            
                            MachineAssignmentQueueEvent event = new MachineAssignmentQueueEvent(task);
                            
                            // Push to queue for later retry
                            queueingDao.produceMachineAssignment(event).setHandler(pushHandler -> {
                                if (pushHandler.succeeded()) {
                                    logger.info(asyncId + " pushed to run-dag queue for machine assignment retry");
                                } else {
                                    logger.error(asyncId + " failed to push to run-dag queue", pushHandler.cause());
                                }
                                future.complete();
                            });
                        } catch (Exception e) {
                            logger.error(asyncId + " error creating machine assignment event", e);
                            future.fail(hostMachineGetHandler.cause());
                        }
                    }
                });
            } else {
                logger.error(asyncId + " couldn't insert in asyncTaskDataStore", asyncTaskInsertHandler.cause());
                future.fail(asyncTaskInsertHandler.cause());
            }
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

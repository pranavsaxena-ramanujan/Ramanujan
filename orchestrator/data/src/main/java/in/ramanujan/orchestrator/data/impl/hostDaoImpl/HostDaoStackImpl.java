package in.ramanujan.orchestrator.data.impl.hostDaoImpl;

import in.ramanujan.orchestrator.base.enums.Status;
import in.ramanujan.orchestrator.base.pojo.AsyncTask;
import in.ramanujan.orchestrator.data.dao.AsyncTaskHostMappingDao;
import in.ramanujan.orchestrator.data.dao.HeartBeatDao;
import in.ramanujan.orchestrator.data.external.OrchestratorApiCaller;
import in.ramanujan.orchestrator.data.dao.HostsDao;
import in.ramanujan.orchestrator.data.dao.StorageDao;
import io.vertx.core.CompositeFuture;
import io.vertx.core.Future;
import io.vertx.core.logging.Logger;
import io.vertx.core.logging.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.*;

@Component
public class HostDaoStackImpl implements HostsDao {

    Logger logger= LoggerFactory.getLogger(HostDaoStackImpl.class);

    private Stack<String> hostStack;

    private Set<String> isHostInStack;
    private Map<String, String> hostClusters;
    private Map<String, Long> hostLastSeen;
    private Map<String, Integer> hostAffinityLimits;
    static final long HOST_STALE_MILLIS = 60_000L;

    @Autowired
    private AsyncTaskHostMappingDao asyncTaskHostMappingDao;

    @Autowired
    private HeartBeatDao heartBeatDao;

    @Autowired
    private OrchestratorApiCaller orchestratorApiCaller;

    @Autowired
    private StorageDao storageDao;

    @Autowired
    private in.ramanujan.orchestrator.data.dao.NativeAffinityDao nativeAffinityDao;

    @Autowired
    private in.ramanujan.orchestrator.data.dao.NativeDeliveryDao nativeDeliveryDao;
    @Autowired private in.ramanujan.orchestrator.data.dao.CapacityDao capacityDao;

    @PostConstruct
    public void init() {
        hostStack = new Stack<>();
        isHostInStack = new HashSet<>();
        hostClusters = new HashMap<>();
        hostLastSeen = new HashMap<>();
        hostAffinityLimits = new HashMap<>();
    }

    @Override
    public synchronized Future<String> getMachine(AsyncTask asyncTask, Boolean resumeComputation) {
        if (asyncTask.getLlm() == null || nativeAffinityDao == null) {
            return reservedHosts(asyncTask).compose(reserved ->
                    selectMachine(asyncTask, resumeComputation, Collections.emptyMap(), reserved));
        }
        Map<String, List<in.ramanujan.db.layer.schema.NativeAffinityOwner>> owners =
                new java.util.concurrent.ConcurrentHashMap<>();
        List<Future> queries = new ArrayList<>();
        Future<Set<String>> reserved = reservedHosts(asyncTask);
        queries.add(reserved);
        for (String host : new ArrayList<>(hostStack)) {
            if (asyncTask.getClusterId() != null && !asyncTask.getClusterId().equals(hostClusters.get(host))) continue;
            if (asyncTask.getNativePreferredHost() != null && !asyncTask.getNativePreferredHost().equals(host)) continue;
            queries.add(nativeAffinityDao.forHost(host).map(rows -> {
                owners.put(host, rows);
                return null;
            }));
        }
        Future<String> result = Future.future();
        CompositeFuture.all(queries).setHandler(handler -> {
            if (handler.failed()) result.fail(handler.cause());
            else selectMachine(asyncTask, resumeComputation, owners, reserved.result()).setHandler(selected -> {
                if (selected.succeeded()) result.complete(selected.result()); else result.fail(selected.cause());
            });
        });
        return result;
    }

    private Future<Set<String>> reservedHosts(AsyncTask task) {
        if (capacityDao == null || !capacityDao.enabled() || task.getNativePreferredHost() != null) {
            return Future.succeededFuture(Collections.emptySet());
        }
        Set<String> clusters = task.getClusterId() == null
                ? new HashSet<>(hostClusters.values()) : Collections.singleton(task.getClusterId());
        Set<String> reserved = new HashSet<>();
        Future<Void> result = Future.succeededFuture();
        for (String cluster : clusters) if (cluster != null) {
            result = result.compose(ignored -> capacityDao.reservedHosts(cluster).map(hosts -> {
                reserved.addAll(hosts);
                return (Void) null;
            }));
        }
        return result.map(ignored -> reserved);
    }

    private synchronized Future<String> selectMachine(AsyncTask asyncTask, Boolean resumeComputation,
            Map<String, List<in.ramanujan.db.layer.schema.NativeAffinityOwner>> owners) {
        return selectMachine(asyncTask, resumeComputation, owners, Collections.emptySet());
    }

    private synchronized Future<String> selectMachine(AsyncTask asyncTask, Boolean resumeComputation,
            Map<String, List<in.ramanujan.db.layer.schema.NativeAffinityOwner>> owners, Set<String> reservedHosts) {
        String candidate = null;
        int candidateIndex = -1;
        int candidateOwned = Integer.MAX_VALUE;
        long now = System.currentTimeMillis();
        for (int index = hostStack.size() - 1; index >= 0; index--) {
            String host = hostStack.get(index);
            if (now - hostLastSeen.getOrDefault(host, 0L) >= HOST_STALE_MILLIS) {
                hostStack.remove(index);
                isHostInStack.remove(host);
                hostClusters.remove(host);
                hostLastSeen.remove(host);
                hostAffinityLimits.remove(host);
                if (candidateIndex > index) candidateIndex--;
            } else if ((asyncTask.getClusterId() == null || asyncTask.getClusterId().equals(hostClusters.get(host)))
                    && (asyncTask.getNativePreferredHost() == null || asyncTask.getNativePreferredHost().equals(host))) {
                if (reservedHosts.contains(host)) continue;
                Set<String> active = new HashSet<>();
                for (in.ramanujan.db.layer.schema.NativeAffinityOwner owner :
                        owners.getOrDefault(host, Collections.emptyList())) {
                    if ("ACTIVE".equals(owner.getState()) && owner.getLastUsed() != null
                            && now - owner.getLastUsed() < in.ramanujan.orchestrator.data.dao.NativeAffinityDao.LEASE_MILLIS) {
                        active.add(owner.getBindingId());
                    }
                }
                int newBindings = 0;
                if (asyncTask.getNativeBindings() != null) {
                    for (String binding : asyncTask.getNativeBindings()) if (!active.contains(binding)) newBindings++;
                }
                int limit = hostAffinityLimits.getOrDefault(host, Integer.MAX_VALUE);
                if (limit == Integer.MAX_VALUE && (asyncTask.getLlm() == null || asyncTask.getLlm().get("planId") == null)) limit = 8;
                if (asyncTask.getLlm() != null && newBindings > 0 && active.size() + newBindings > limit) continue;
                if (candidate == null || active.size() < candidateOwned) {
                    candidate = host;
                    candidateIndex = index;
                    candidateOwned = active.size();
                }
            }
        }
        if(candidate == null) {
            logger.error(asyncTask.getUuid() + " no machine available for computation");
            return Future.succeededFuture("No Machine available");
        } else {
            String hostId = candidate;
            hostStack.remove(candidateIndex);
            isHostInStack.remove(hostId);
            asyncTask.setAssignedCluster(hostClusters.get(hostId));
            Future<String> future = Future.future();

            logger.info(asyncTask.getUuid() + " has got probable machine " + hostId);
            //This table has a uniqueId on hostId. This will fail if more than one process tries to take it.
            asyncTaskHostMappingDao.createMapping(asyncTask, hostId, resumeComputation).setHandler(mappingCreateHandler -> {
               if(mappingCreateHandler.succeeded()) {
                   logger.info(asyncTask.getUuid() + " has been assigned machine " + hostId);
                   future.complete(hostId);
               } else {
                   removeFromStack(hostId);
                   logger.error(asyncTask.getUuid() + " has NOT been assigned machine " + hostId, mappingCreateHandler.cause());
                   //on taskStatus, if host is null, reassign will happen.
                   future.complete();
               }
            });

            return future;
        }
    }

    @Override
    public Future<AsyncTask> putMachineForComputation(String hostId) {
        return putMachineForComputation(hostId, null);
    }

    @Override
    public synchronized Future<AsyncTask> putMachineForComputation(String hostId, String clusterId) {
        return putMachineForComputation(hostId, clusterId, Integer.MAX_VALUE);
    }

    @Override
    public synchronized Future<AsyncTask> putMachineForComputation(String hostId, String clusterId, int affinityLimit) {
        if (hostId == null || hostId.isEmpty()) return Future.failedFuture("uuid is required");
        hostClusters.put(hostId, clusterId);
        hostLastSeen.put(hostId, System.currentTimeMillis());
        hostAffinityLimits.put(hostId, Math.max(0, affinityLimit));
        /*
        * Check if any asynctask is being processed by the hostId
        * Check if there is an entry in availableHost with proper timelimit and status as ENGAGED.
        * If not, it can be added to the stack.
        * Also, check if the stack size is above 10^4. If yes, then send this request to some other host and start removing stack items.
        * */
        Future<AsyncTask> future = Future.future();
        if(hostStack.size() >= 10000) {
            callOrchestratorApiForMachineAddition(hostId, clusterId, affinityLimit, future);
            removeOldHosts();
            return future;
        }
        //logger.info("stack size: " + hostStack.size());
        asyncTaskHostMappingDao.getMapping(hostId).setHandler(handler -> {
           if(handler.succeeded()) {
               if(handler.result() == null) {
                   addInStack(hostId, future);
               } else if (Status.FAILURE.getKeyName().equalsIgnoreCase(handler.result().getStatus())) {
                   // hostMapping is unique per host, so a failed task's row would block the next assignment.
                   String failedTask = handler.result().getUuid();
                   Future<Void> removed = asyncTaskHostMappingDao.removeMapping(hostId, failedTask);
                   if (removed == null) {
                       addInStack(hostId, future);
                       return;
                   }
                   removed.setHandler(cleanup -> {
                       if (cleanup.failed()) logger.error("could not clear failed task " + failedTask + " from " + hostId, cleanup.cause());
                       addInStack(hostId, future);
                   });
               } else {
                   logger.info("Machine " + hostId + " is assigned " + handler.result().getUuid());
                   removeFromStack(hostId);
                   AsyncTask asyncTask = handler.result();
                   if (!Objects.equals(clusterId, asyncTask.getAssignedCluster())
                           || (asyncTask.getClusterId() != null && !asyncTask.getClusterId().equals(clusterId))) {
                       future.complete();
                       return;
                   }
                   if (asyncTask != null) {
                       if (asyncTask.getLlm() != null) {
                           if ("ASSIGNED".equals(asyncTask.getNativeState())) {
                               nativeDeliveryDao.claim(asyncTask).setHandler(delivered -> {
                                   if (delivered.failed()) future.fail(delivered.cause());
                                   else future.complete(Boolean.TRUE.equals(delivered.result()) ? asyncTask : null);
                               });
                           } else future.complete();
                           return;
                       }
                       if (asyncTask.getClusterId() != null && asyncTask.getAssignedNonce() != null) {
                           nativeDeliveryDao.claim(asyncTask).setHandler(delivered -> {
                               if (delivered.failed()) future.fail(delivered.cause());
                               else if (!Boolean.TRUE.equals(delivered.result())) future.complete();
                               else populateTask(asyncTask, hostId, future);
                           });
                       } else populateTask(asyncTask, hostId, future);
                   } else {
                       future.complete(handler.result());
                   }
               }
           } else {
               logger.error("machine " + hostId + " not able to get current asyncTask mapping due to ", handler.cause());
               future.fail(handler.cause());
           }
        });
        return future;
    }

    private void populateTask(AsyncTask asyncTask, String hostId, Future<AsyncTask> future) {
        CompositeFuture.all(getAsyncTaskRuleEngineInputStorageDao(asyncTask),
                getCheckpointData(asyncTask), getDebugBreakpoints(asyncTask)).setHandler(compositeHandler -> {
            if (compositeHandler.succeeded()) future.complete(asyncTask);
            else future.fail(compositeHandler.cause());
        });
    }

    private Future<Void> getDebugBreakpoints(AsyncTask asyncTask) {
        Future<Void> future = Future.future();
        storageDao.getBreakpoints(asyncTask.getUuid()).setHandler(handler -> {
           if(handler.succeeded()) {
               if(handler.result() != null) {
                   asyncTask.setBreakpoints(handler.result().getLines());
               } else {
                   asyncTask.setBreakpoints(new ArrayList<>());
               }
               future.complete();
           } else {
               future.fail(handler.cause());
           }
        });
        return future;
    }

    private Future<Void> getCheckpointData(AsyncTask asyncTask) {
        Future<Void> future = Future.future();
        if(asyncTask.getCheckpoint() == null) {
            return Future.succeededFuture();
        }
        try {
            storageDao.getCheckpoint(asyncTask.getUuid()).setHandler(handler -> {
                if(handler.succeeded()) {
                    asyncTask.setCheckpoint(handler.result());
                } else {
                    asyncTask.setCheckpoint(null);
                }
                future.complete();
            });
        } catch (Exception e) {
            asyncTask.setCheckpoint(null);
        }
        return future;
    }

    private Future<Void> getAsyncTaskRuleEngineInputStorageDao(AsyncTask asyncTask) {
        Future<Void> future = Future.future();
        try {
            storageDao.getAsyncTaskRuleEngineInput(asyncTask.getUuid()).setHandler(ruleEngineInputHandler -> {
               if(ruleEngineInputHandler.succeeded()) {
                   asyncTask.setRuleEngineInput(ruleEngineInputHandler.result());
                   future.complete();
               } else {
                   future.fail(ruleEngineInputHandler.cause());
               }
            });
        } catch (Exception e) {
            future.fail(e);
        }
        return future;
    }

    private synchronized void removeOldHosts() {
        hostStack = new Stack<>();
        isHostInStack = new HashSet<>();
        hostClusters.clear();
        hostLastSeen.clear();
        hostAffinityLimits.clear();
    }

    private void callOrchestratorApiForMachineAddition(String hostId, String clusterId, int affinityLimit, Future<AsyncTask> future) {
        Future<Void> forwarded = affinityLimit == Integer.MAX_VALUE
                ? orchestratorApiCaller.callOpenPingApiWithRetry(hostId, 3, clusterId)
                : orchestratorApiCaller.callOpenPingApiWithRetry(hostId, 3, clusterId, affinityLimit);
        forwarded.setHandler(handler -> {
           if(handler.succeeded()) {
               future.complete();
           } else {
               future.fail(handler.cause());
           }
        });
    }

    private synchronized void removeFromStack(String hostId) {
        isHostInStack.remove(hostId);
        hostStack.remove(hostId);
    }

    private synchronized void addInStack(String hostId, Future<AsyncTask> future) {
        if(!isHostInStack.contains(hostId)) {
            hostStack.push(hostId);
            isHostInStack.add(hostId);
            logger.info("pushed in stack");
        }
        future.complete(null);
    }
}

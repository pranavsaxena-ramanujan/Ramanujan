package in.ramanujan.orchestrator.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import in.ramanujan.orchestrator.base.enums.AsyncTaskFields;
import in.ramanujan.orchestrator.base.enums.Status;
import in.ramanujan.orchestrator.base.pojo.AsyncTask;
import in.ramanujan.orchestrator.base.pojo.HostResult;
import in.ramanujan.orchestrator.data.dao.AsyncTaskDao;
import in.ramanujan.orchestrator.data.dao.AsyncTaskHostMappingDao;
import in.ramanujan.orchestrator.data.dao.StorageDao;
import io.vertx.core.Future;
import io.vertx.core.json.JsonObject;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.List;
import java.util.Objects;

@Component
public class TaskCompleteService {

    @Autowired
    private AsyncTaskHostMappingDao asyncTaskHostMappingDao;

    @Autowired
    private AsyncTaskDao asyncTaskDao;

    @Autowired
    private StorageDao storageDao;

    @Autowired
    private NativeLlmService nativeLlmService;

    private final ObjectMapper objectMapper = new ObjectMapper();

    public Future<Void> completeTask(String hostId, String uuid, Object resultOfComputation) {
        return completeTask(hostId, uuid, resultOfComputation, null);
    }

    public Future<Void> completeTask(String hostId, String uuid, Object resultOfComputation, String clusterId) {
        return completeTask(hostId, uuid, resultOfComputation, clusterId, null);
    }

    public Future<Void> completeTask(String hostId, String uuid, Object resultOfComputation, String clusterId, String error) {
        Future<Void> future = Future.future();
        if (hostId == null || uuid == null) return Future.failedFuture(new SecurityException("Task assignment is required"));
        asyncTaskHostMappingDao.getMapping(hostId).setHandler(assignment -> {
            if (assignment.failed()) {
                future.fail(assignment.cause());
                return;
            }
            AsyncTask task = assignment.result();
            if (task == null || !uuid.equals(task.getUuid())
                    || !Objects.equals(clusterId, task.getAssignedCluster())
                    || (task.getClusterId() != null && !task.getClusterId().equals(clusterId))) {
                future.fail(new SecurityException("Wrong task assignment"));
                return;
            }
            if (task.getLlm() != null) {
                Map<String, Object> payload = new java.util.LinkedHashMap<>();
                payload.put("uuid", uuid);
                payload.put("hostId", hostId);
                payload.put("clusterId", clusterId);
                payload.put("data", resultOfComputation);
                if (error != null) payload.put("error", error);
                nativeLlmService.complete(task, payload).setHandler(completed -> {
                    if (completed.succeeded()) future.complete(); else future.fail(completed.cause());
                });
                return;
            }

        mergeBinaryOutputs(task, (Map<String, Object>) resultOfComputation).compose(results -> {
            HostResult hostResult = new HostResult();
            hostResult.setMap(results);
            return refreshVariables(uuid, hostResult);
        }).setHandler(refreshVariableHandler -> {
            if(refreshVariableHandler.succeeded()) {
                updateAsyncTask(uuid, hostId, future);
            } else {
                future.fail(refreshVariableHandler.cause());
            }
        });
        });

        return future;
    }

    private Future<Map<String, Object>> mergeBinaryOutputs(AsyncTask task, Map<String, Object> results) {
        if (task.getBinaryArrayFiles() == null || task.getBinaryArrayFiles().isEmpty()) return Future.succeededFuture(results);
        try {
            return storageDao.getAsyncTaskRuleEngineInput(task.getUuid()).compose(input -> {
                Future<Map<String, Object>> merged = Future.future();
                io.vertx.core.Handler<io.vertx.core.Promise<Map<String, Object>>> merge = blocking -> {
                    try {
                        Map<String, Object> copy = new java.util.LinkedHashMap<>(results);
                        Map<String, Map<String, Object>> arrays = copy.get("arrayIndex") instanceof Map
                                ? new java.util.LinkedHashMap<>((Map<String, Map<String, Object>>) copy.get("arrayIndex"))
                                : new java.util.LinkedHashMap<>();
                        for (in.ramanujan.pojo.ruleEngineInputUnitsExt.array.Array array : input.getArrays()) {
                            String path = task.getBinaryArrayFiles().get(array.getId());
                            if (path == null) continue;
                            Map<String, Object> values = new java.util.LinkedHashMap<>();
                            try (java.io.DataInputStream file = new java.io.DataInputStream(
                                    new java.io.BufferedInputStream(new java.io.FileInputStream(path)))) {
                                long count = new java.io.File(path).length() / Float.BYTES;
                                for (long offset = 0; offset < count; offset++) {
                                    long position = offset;
                                    List<Integer> dimensions = array.getDimension();
                                    String index = Long.toString(offset);
                                    if (dimensions != null && dimensions.size() > 1) {
                                        List<String> parts = new java.util.ArrayList<>();
                                        for (int d = dimensions.size() - 1; d >= 0; d--) {
                                            int size = dimensions.get(d);
                                            if (size <= 0) throw new IllegalArgumentException("Invalid array dimension");
                                            parts.add(0, Long.toString(position % size));
                                            position /= size;
                                        }
                                        index = String.join("_", parts);
                                    }
                                    values.put(index, Float.intBitsToFloat(Integer.reverseBytes(file.readInt())));
                                }
                            }
                            arrays.put(array.getId(), values);
                        }
                        copy.put("arrayIndex", arrays);
                        blocking.complete(copy);
                    } catch (Exception ex) { blocking.fail(ex); }
                };
                io.vertx.core.Context context = io.vertx.core.Vertx.currentContext();
                if (context == null) {
                    io.vertx.core.Promise<Map<String, Object>> immediate = io.vertx.core.Promise.promise();
                    merge.handle(immediate);
                    return immediate.future();
                }
                context.<Map<String, Object>>executeBlocking(merge, false, handler -> {
                    if (handler.succeeded()) merged.complete(handler.result()); else merged.fail(handler.cause());
                });
                return merged;
            });
        } catch (Exception ex) { return Future.failedFuture(ex); }
    }

    private Future<Object> refreshVariables(String uuid, HostResult hostResult) {
        Future<Object> future = Future.future();
        try {
            storageDao.storeAsyncTaskResult(uuid, JsonObject.mapFrom(hostResult).toString()).setHandler(handler -> {
                if (handler.succeeded()) {
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

    private void updateAsyncTask(String uuid, String hostId, Future<Void> future) {
        JsonObject updateQuery = new JsonObject()
                .put(AsyncTaskFields.status.getFieldName(), Status.SUCCESS.getKeyName());
        asyncTaskDao.update(uuid, updateQuery.getMap()).setHandler(updateAsyncTaskHandler -> {
            if(updateAsyncTaskHandler.succeeded())  {
                asyncTaskHostMappingDao.removeMapping(hostId, uuid).setHandler(taskRemoveHandler -> {
                    if(taskRemoveHandler.succeeded()) {
                        future.complete();
                    } else {
                        future.fail(taskRemoveHandler.cause());
                    }
                });
            } else {
                future.fail(updateAsyncTaskHandler.cause());
            }
        });
    }
}

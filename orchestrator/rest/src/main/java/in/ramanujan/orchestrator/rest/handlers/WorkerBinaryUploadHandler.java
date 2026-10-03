package in.ramanujan.orchestrator.rest.handlers;

import in.ramanujan.orchestrator.base.pojo.AsyncTask;
import in.ramanujan.orchestrator.data.dao.AsyncTaskDao;
import in.ramanujan.orchestrator.data.dao.AsyncTaskHostMappingDao;
import in.ramanujan.orchestrator.data.dao.StorageDao;
import io.vertx.core.Future;
import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.RoutingContext;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.io.File;
import java.util.*;

@Component
public class WorkerBinaryUploadHandler {
    @Autowired private AsyncTaskHostMappingDao mappings;
    @Autowired private AsyncTaskDao tasks;
    @Autowired private StorageDao storage;

    public void handle(RoutingContext event) {
        String taskId = event.queryParams().get("taskUuid");
        String host = taskId == null ? event.queryParams().get("hostId") : event.queryParams().get("uuid");
        if (taskId == null) taskId = event.queryParams().get("uuid");
        String cluster = event.queryParams().get("clusterId");
        String arrayId = event.queryParams().get("arrayId");
        if (taskId == null || host == null || arrayId == null || event.getBody() == null) {
            respond(event, 400, "taskUuid, device uuid, arrayId and binary body are required");
            return;
        }
        String expectedTask = taskId;
        mappings.getMapping(host).compose(task -> {
            if (task == null || !expectedTask.equals(task.getUuid()) || !host.equals(task.getHostAssigned())
                    || !Objects.equals(cluster, task.getAssignedCluster()) || task.getLlm() != null
                    || (task.getClusterId() != null && !task.getClusterId().equals(cluster))) {
                return Future.<AsyncTask>failedFuture(new SecurityException("Wrong task assignment"));
            }
            try {
                return storage.getAsyncTaskRuleEngineInput(task.getUuid()).compose(input -> {
                    if (input == null || input.getArrays() == null || input.getArrays().stream()
                            .noneMatch(array -> arrayId.equals(array.getId()))) {
                        return Future.failedFuture(new SecurityException("Array is not part of assigned task"));
                    }
                    return Future.succeededFuture(task);
                });
            } catch (Exception ex) { return Future.failedFuture(ex); }
        }).compose(task -> {
            Future<String> written = Future.future();
            String directory = ".ramanujan-central-outputs";
            String path = new File(directory, UUID.randomUUID().toString() + ".bin").getAbsolutePath();
            event.vertx().fileSystem().mkdirs(directory, created -> {
                if (created.failed()) written.fail(created.cause());
                else event.vertx().fileSystem().writeFile(path, event.getBody(), saved -> {
                    if (saved.succeeded()) written.complete(path); else written.fail(saved.cause());
                });
            });
            return written.compose(pathWritten -> {
                return mappings.getMapping(host).compose(current -> {
                if (current == null || !task.getUuid().equals(current.getUuid())
                        || !Objects.equals(task.getAssignedNonce(), current.getAssignedNonce())
                        || !Objects.equals(cluster, current.getAssignedCluster())) {
                    event.vertx().fileSystem().delete(pathWritten, ignored -> {});
                    return Future.failedFuture(new SecurityException("Task assignment changed during upload"));
                }
                Map<String, String> outputs = new LinkedHashMap<>();
                if (current.getBinaryArrayFiles() != null) outputs.putAll(current.getBinaryArrayFiles());
                outputs.put(arrayId, pathWritten);
                return tasks.update(task.getUuid(), Collections.singletonMap("binaryArrayFiles", outputs)).recover(error -> {
                    event.vertx().fileSystem().delete(pathWritten, ignored -> {});
                    return Future.failedFuture(error);
                });
                });
            });
        }).setHandler(saved -> {
            if (saved.succeeded()) respond(event, 200, null);
            else respond(event, saved.cause() instanceof SecurityException ? 403 : 500, saved.cause().getMessage());
        });
    }

    private static void respond(RoutingContext event, int status, String message) {
        JsonObject response = new JsonObject().put("status", status == 200 ? "SUCCESS" : "ERROR");
        if (message != null) response.put("message", message);
        event.response().setStatusCode(status).putHeader("Content-Type", "application/json").end(response.encode());
    }
}

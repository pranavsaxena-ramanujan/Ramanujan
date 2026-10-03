package in.ramanujan.orchestrator.rest.handlers;

import in.ramanujan.orchestrator.base.pojo.AsyncTask;
import in.ramanujan.orchestrator.data.dao.AsyncTaskHostMappingDao;
import in.ramanujan.orchestrator.data.dao.StorageDao;
import io.vertx.core.Future;
import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.RoutingContext;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.io.File;
import java.util.Objects;

@Component
public class WorkerBinaryHandler {
    @Autowired private AsyncTaskHostMappingDao mappings;
    @Autowired private StorageDao storage;

    public void handle(RoutingContext event, boolean stat) {
        String host = event.queryParams().get("uuid");
        String cluster = event.queryParams().get("clusterId");
        String path = event.queryParams().get("path");
        if (host == null || path == null) {
            reject(event, 400, "uuid and path are required");
            return;
        }
        mappings.getMapping(host).compose(task -> {
            if (task == null || !host.equals(task.getHostAssigned())
                    || !Objects.equals(cluster, task.getAssignedCluster())
                    || (task.getClusterId() != null && !task.getClusterId().equals(cluster))) {
                return Future.<Boolean>failedFuture(new SecurityException("Wrong task assignment"));
            }
            if (task.getLlm() != null) {
                return Future.succeededFuture("ASSIGNED".equals(task.getNativeState())
                        && task.getNativeFiles() != null && task.getNativeFiles().contains(path));
            }
            try {
                return storage.getAsyncTaskRuleEngineInput(task.getUuid()).map(input -> {
                    if (input == null || input.getArrays() == null) return false;
                    return input.getArrays().stream().anyMatch(array -> path.equals(array.getBinaryFile()));
                });
            } catch (Exception ex) { return Future.failedFuture(ex); }
        }).setHandler(allowed -> {
            if (allowed.failed() || !Boolean.TRUE.equals(allowed.result())) {
                reject(event, 403, "Path is not authorized for this assignment");
                return;
            }
            File file = new File(path);
            if (!file.isFile()) {
                reject(event, 404, "File not found");
            } else if (stat) {
                event.response().putHeader("Content-Type", "application/json").end(new JsonObject()
                        .put("status", "SUCCESS").put("size", file.length()).put("mtime", file.lastModified()).encode());
            } else {
                event.response().putHeader("Content-Type", "application/octet-stream")
                        .putHeader("X-Ramanujan-Mtime", String.valueOf(file.lastModified())).sendFile(file.getAbsolutePath());
            }
        });
    }

    private static void reject(RoutingContext event, int status, String message) {
        event.response().setStatusCode(status).putHeader("Content-Type", "application/json")
                .end(new JsonObject().put("status", "ERROR").put("message", message).encode());
    }
}

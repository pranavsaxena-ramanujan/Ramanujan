package in.ramanujan.orchestrator.rest.handlers;

import in.ramanujan.orchestrator.data.dao.CapacityDao;
import io.vertx.core.Future;
import io.vertx.core.json.JsonObject;
import io.vertx.core.logging.Logger;
import io.vertx.core.logging.LoggerFactory;
import io.vertx.ext.web.RoutingContext;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;

@Component
public class CapacityHandler {
    @Autowired private CapacityDao capacity;
    private final Logger logger = LoggerFactory.getLogger(CapacityHandler.class);

    public void handle(RoutingContext event, String operation) {
        if (!capacity.enabled()) {
            event.response().setStatusCode("report".equals(operation) ? 202 : 503)
                    .putHeader("Content-Type", "application/json")
                    .end(new JsonObject().put("status", "DISABLED").put("accepted", false)
                            .put("error", "Capacity-aware placement is disabled").encode());
            return;
        }
        try {
            if ("snapshots".equals(operation)) {
                String cluster = event.request().getParam("clusterId");
                if (cluster == null || cluster.isEmpty() || cluster.length() > 100) {
                    throw new IllegalArgumentException("clusterId is required");
                }
                respond(event, capacity.snapshots(cluster).map(hosts -> {
                    Map<String, Object> result = new LinkedHashMap<>();
                    result.put("hosts", hosts);
                    return result;
                }));
                return;
            }
            JsonObject body = event.getBodyAsJson();
            if (body == null) throw new IllegalArgumentException("JSON body required");
            Map<String, Object> request = body.getMap();
            Future<Map<String, Object>> result;
            if ("report".equals(operation)) result = capacity.report(request);
            else if ("plan".equals(operation)) result = capacity.reserve(request);
            else {
                Object id = request.get("planId");
                if (!(id instanceof String) || ((String) id).length() != 36) throw new IllegalArgumentException("planId required");
                result = capacity.release(CapacityDao.cluster(request), (String) id).map(ignored -> new LinkedHashMap<>());
            }
            respond(event, result);
        } catch (Exception error) { respond(event, Future.failedFuture(error)); }
    }

    private void respond(RoutingContext event, Future<Map<String, Object>> result) {
        result.setHandler(done -> {
            JsonObject response = new JsonObject().put("status", done.succeeded() ? "SUCCESS" : "ERROR");
            int status = 200;
            if (done.succeeded()) response.mergeIn(new JsonObject(done.result()));
            else {
                logger.warn("Capacity request rejected: " + done.cause().getMessage());
                status = done.cause() instanceof IllegalArgumentException ? 400
                        : done.cause() instanceof IllegalStateException ? 409 : 500;
                response.put("error", done.cause().getMessage());
            }
            event.response().setStatusCode(status).putHeader("Content-Type", "application/json").end(response.encode());
        });
    }
}

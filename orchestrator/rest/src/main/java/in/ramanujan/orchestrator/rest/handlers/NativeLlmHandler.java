package in.ramanujan.orchestrator.rest.handlers;

import in.ramanujan.orchestrator.service.NativeLlmService;
import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.RoutingContext;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;

@Component
public class NativeLlmHandler {
    @Autowired private NativeLlmService service;

    @SuppressWarnings("unchecked")
    public void handle(RoutingContext event, String op) {
        JsonObject body = event.getBodyAsJson();
        if (body == null) {
            event.response().setStatusCode(400).end(new JsonObject().put("status", "ERROR").put("error", "JSON body required").encode());
            return;
        }
        service.execute(body.getMap(), op, event.vertx()).setHandler(completed -> {
            Map<String, Object> response = new LinkedHashMap<>();
            int status = 200;
            if (completed.failed()) {
                status = completed.cause() instanceof IllegalArgumentException ? 400 : 500;
                response.put("status", "ERROR");
                response.put("error", completed.cause().getMessage());
            } else {
                Map<String, Object> payload = completed.result();
                if (payload.get("error") != null) {
                    status = Boolean.TRUE.equals(payload.get("timedOut")) ? 504 : 500;
                    response.put("status", "ERROR");
                    response.put("error", payload.get("error"));
                    response.put("worker", payload.get("hostId"));
                } else if ("SUCCESS".equals(payload.get("status"))) {
                    response.putAll(payload);
                } else {
                    response.put("status", "SUCCESS");
                    if (payload.get("data") instanceof Map) response.putAll((Map<String, Object>) payload.get("data"));
                    response.put("worker", payload.get("hostId"));
                }
            }
            event.response().setStatusCode(status).putHeader("Content-Type", "application/json").end(new JsonObject(response).encode());
        });
    }
}

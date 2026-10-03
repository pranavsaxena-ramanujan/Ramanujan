package in.ramanujan.orchestrator.data.external;

import in.ramanujan.orchestrator.base.configuration.ConfigKey;
import in.ramanujan.orchestrator.base.configuration.ConfigurationGetter;
import io.vertx.core.Future;
import io.vertx.core.Vertx;
import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.client.WebClient;
import io.vertx.ext.web.client.WebClientOptions;
import org.springframework.stereotype.Component;

@Component
public class OrchestratorApiCaller {

    private WebClient webClient;

    private String openPingUri = "/pings/open";

    private WebClient getWebClient() {
        if(webClient == null) {
            webClient = WebClient.create(
                    Vertx.vertx(),
                    new WebClientOptions()
                            .setDefaultHost(ConfigurationGetter.getString(ConfigKey.ORCHESTRATOR_HOST_KEY))
                            .setDefaultPort(ConfigurationGetter.getInt(ConfigKey.ORCHESTRATOR_PORT_KEY))
            );
        }
        return webClient;
    }

    public void setWebClient(String ip, int port) {
        webClient = WebClient.create(
                Vertx.vertx(),
                new WebClientOptions()
                        .setDefaultHost(ip)
                        .setDefaultPort(port)
        );
    }

    public Future<Void> callOpenPingApiWithRetry(final String hostId, final Integer retryNumber) {
        return callOpenPingApiWithRetry(hostId, retryNumber, null);
    }

    public Future<Void> callOpenPingApiWithRetry(final String hostId, final Integer retryNumber,
                                                final String clusterId) {
        return callOpenPingApiWithRetry(hostId, retryNumber, clusterId, Integer.MAX_VALUE);
    }

    public Future<Void> callOpenPingApiWithRetry(final String hostId, final Integer retryNumber,
                                                final String clusterId, final int affinityLimit) {
        Future<Void> future = Future.future();
        callOpenPingApi(hostId, clusterId, affinityLimit).setHandler(handler -> {
            if(handler.succeeded()){
                future.complete();
            } else {
                if(retryNumber == 0) {
                    future.fail(handler.cause());
                } else {
                    callOpenPingApiWithRetry(hostId, retryNumber - 1, clusterId, affinityLimit).setHandler(retryHandler -> {
                       if(retryHandler.succeeded()) {
                           future.complete();
                       } else {
                           future.fail(retryHandler.cause());
                       }
                    });
                }
            }
        });
        return future;
    }

    private Future<Void> callOpenPingApi(final String hostId, final String clusterId, final int affinityLimit) {
        Future<Void> future = Future.future();
        io.vertx.ext.web.client.HttpRequest<io.vertx.core.buffer.Buffer> request =
                getWebClient().post(openPingUri).addQueryParam("uuid", hostId);
        if (clusterId != null) request.addQueryParam("clusterId", clusterId);
        if (affinityLimit != Integer.MAX_VALUE) request.addQueryParam("affinityLimit", Integer.toString(affinityLimit));
        request.sendJsonObject(new JsonObject(), handler -> {
            if(handler.succeeded() && handler.result().statusCode() == 200) {
                future.complete();
            } else {
                future.fail(handler.failed() ? handler.cause() :
                        new IllegalStateException("Open ping forwarding failed: " + handler.result().statusCode()));
            }
        });
        return future;
    }
}

package in.ramanujan.orchestrator.data.dao;

import in.ramanujan.db.layer.constants.Keys;
import in.ramanujan.db.layer.enums.QueryType;
import in.ramanujan.db.layer.schema.NativeTaskDelivery;
import in.ramanujan.db.layer.utils.QueryExecutor;
import in.ramanujan.orchestrator.base.pojo.AsyncTask;
import io.vertx.core.Future;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/** A unique durable receipt prevents concurrent polling threads from executing a stateful task twice. */
@Component
public class NativeDeliveryDao {
    @Autowired private QueryExecutor queryExecutor;

    public Future<Boolean> claim(AsyncTask task) {
        Future<Boolean> result = Future.future();
        NativeTaskDelivery receipt = new NativeTaskDelivery();
        receipt.setUuid(task.getLlm() != null ? "N:" + task.getUuid() : "D:" + task.getAssignedNonce());
        receipt.setTaskUuid(task.getUuid());
        receipt.setHostId(task.getHostAssigned());
        receipt.setClusterId(task.getAssignedCluster());
        receipt.setDeliveredAt(System.currentTimeMillis());
        try {
            queryExecutor.execute(receipt, null, QueryType.INSERT).setHandler(inserted -> {
                if (inserted.succeeded()) result.complete(true);
                else {
                    try {
                        queryExecutor.execute(receipt, Keys.UUID, QueryType.SELECT).setHandler(existing -> {
                            if (existing.succeeded() && !existing.result().isEmpty()) result.complete(false);
                            else result.fail(inserted.cause());
                        });
                    } catch (Exception ex) { result.fail(ex); }
                }
            });
        } catch (Exception ex) { result.fail(ex); }
        return result;
    }
}

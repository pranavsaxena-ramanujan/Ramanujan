package in.ramanujan.orchestrator.data.dao;

import in.ramanujan.db.layer.constants.Keys;
import in.ramanujan.db.layer.enums.QueryType;
import in.ramanujan.db.layer.schema.NativeAffinityOwner;
import in.ramanujan.db.layer.schema.NativePositionClaim;
import in.ramanujan.db.layer.utils.QueryExecutor;
import io.vertx.core.Future;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.*;

@Component
public class NativeAffinityDao {
    public static final long LEASE_MILLIS = 31 * 60_000L;
    @Autowired
    private QueryExecutor queryExecutor;
    @Autowired private AsyncTaskDao asyncTaskDao;

    public static String position(Map<String, Object> payload) {
        if ("close".equals(payload.get("op"))) return "close";
        Object value = payload.get("pos");
        if (value == null) return "0";
        if (!(value instanceof Number) || ((Number) value).longValue() < 0
                || ((Number) value).doubleValue() != ((Number) value).longValue()) {
            throw new IllegalArgumentException("pos must be a nonnegative integer");
        }
        return Long.toString(((Number) value).longValue());
    }

    private Future<Void> claimPosition(String binding, String task, String position) {
        NativePositionClaim claim = new NativePositionClaim();
        try {
            byte[] digest = java.security.MessageDigest.getInstance("SHA-256")
                    .digest((binding + ":" + position).getBytes(java.nio.charset.StandardCharsets.UTF_8));
            StringBuilder key = new StringBuilder(64);
            for (byte value : digest) key.append(String.format("%02x", value & 0xff));
            claim.setClaimId(key.toString());
            claim.setTaskUuid(task);
            claim.setBindingId(binding);
            claim.setPosition(position);
            Future<Void> result = Future.future();
            queryExecutor.execute(claim, null, QueryType.INSERT).setHandler(inserted -> {
                if (inserted.succeeded()) result.complete();
                else {
                    try {
                        queryExecutor.execute(claim, Keys.UUID, QueryType.SELECT).setHandler(existing -> {
                            if (existing.succeeded() && !existing.result().isEmpty()
                                    && task.equals(((NativePositionClaim) existing.result().get(0)).getTaskUuid())) result.complete();
                            else result.fail("Stateful position was already submitted; do not replay it");
                        });
                    } catch (Exception ex) { result.fail(ex); }
                }
            });
            return result;
        } catch (Exception ex) { return Future.failedFuture(ex); }
    }

    public Future<NativeAffinityOwner> get(String binding) {
        Future<NativeAffinityOwner> result = Future.future();
        NativeAffinityOwner row = new NativeAffinityOwner();
        row.setBindingId(binding);
        try {
            queryExecutor.execute(row, Keys.UUID, QueryType.SELECT).setHandler(handler -> {
                if (handler.failed()) result.fail(handler.cause());
                else result.complete(handler.result().isEmpty() ? null : (NativeAffinityOwner) handler.result().get(0));
            });
        } catch (Exception ex) { result.fail(ex); }
        return result;
    }

    public Future<List<NativeAffinityOwner>> forHost(String host) {
        Future<List<NativeAffinityOwner>> result = Future.future();
        NativeAffinityOwner row = new NativeAffinityOwner();
        row.setHostId(host);
        row.setState("ACTIVE");
        try {
            queryExecutor.execute(row, "nativeHostState", QueryType.SELECT).setHandler(handler -> {
                if (handler.failed()) result.fail(handler.cause());
                else {
                    List<NativeAffinityOwner> owners = new ArrayList<>();
                    for (Object object : handler.result()) {
                        NativeAffinityOwner owner = (NativeAffinityOwner) object;
                        if ("ACTIVE".equals(owner.getState())) owners.add(owner);
                    }
                    result.complete(owners);
                }
            });
        } catch (Exception ex) { result.fail(ex); }
        return result;
    }

    public Future<Void> claim(List<String> bindings, String host, String cluster, List<String> files,
                              String task, String position) {
        Future<Void> result = Future.succeededFuture();
        for (String binding : bindings) {
            result = result.compose(ignored -> claimOne(binding, host, cluster, files, task, position));
        }
        return result;
    }

    private Future<Void> claimOne(String binding, String host, String cluster, List<String> files, String task, String position) {
        return get(binding).compose(existing -> {
            if (existing != null) {
                if (!host.equals(existing.getHostId()) || !Objects.equals(cluster, existing.getClusterId())
                        || !"ACTIVE".equals(existing.getState())
                        || existing.getLastUsed() == null || System.currentTimeMillis() - existing.getLastUsed() >= LEASE_MILLIS) {
                    return Future.failedFuture("Native session is unavailable or owned by another device");
                }
                if (existing.getLastTask() != null && !task.equals(existing.getLastTask())) {
                    return asyncTaskDao.getAsyncTask(existing.getLastTask()).compose(previous -> {
                        if (previous == null || !"COMPLETED".equals(previous.getNativeState())) {
                            return Future.failedFuture("Previous native task is unresolved; do not advance or replay the session");
                        }
                        return refreshOwner(binding, files, task, position);
                    });
                }
                return refreshOwner(binding, files, task, position);
            }
            NativeAffinityOwner row = new NativeAffinityOwner();
            row.setBindingId(binding);
            row.setHostId(host);
            row.setClusterId(cluster);
            row.setState("ACTIVE");
            row.setLastUsed(System.currentTimeMillis());
            row.setLastTask(task);
            row.setLastPosition(position);
            try {
                row.setFiles(new com.fasterxml.jackson.databind.ObjectMapper().writeValueAsString(files));
                Future<Void> result = Future.future();
                queryExecutor.execute(row, null, QueryType.INSERT).setHandler(handler -> {
                    if (handler.succeeded()) claimPosition(binding, task, position).setHandler(claimed -> {
                        if (claimed.succeeded()) result.complete(); else result.fail(claimed.cause());
                    });
                    else get(binding).setHandler(owner -> {
                        if (owner.succeeded() && owner.result() != null && host.equals(owner.result().getHostId())
                                && Objects.equals(cluster, owner.result().getClusterId())
                                && "ACTIVE".equals(owner.result().getState())) {
                            claimOne(binding, host, cluster, files, task, position).setHandler(claimed -> {
                                if (claimed.succeeded()) result.complete(); else result.fail(claimed.cause());
                            });
                        }
                        else result.fail(handler.cause());
                    });
                });
                return result;
            } catch (Exception ex) { return Future.failedFuture(ex); }
        });
    }

    private Future<Void> refreshOwner(String binding, List<String> files, String task, String position) {
        NativeAffinityOwner refresh = new NativeAffinityOwner();
        refresh.setBindingId(binding);
        refresh.setLastUsed(System.currentTimeMillis());
        refresh.setLastTask(task);
        refresh.setLastPosition(position);
        Future<Void> result = Future.future();
        try {
            refresh.setFiles(new com.fasterxml.jackson.databind.ObjectMapper().writeValueAsString(files));
            claimPosition(binding, task, position).setHandler(claimed -> {
                if (claimed.failed()) { result.fail(claimed.cause()); return; }
                try {
                    queryExecutor.execute(refresh, Keys.UUID, QueryType.UPDATE).setHandler(updated -> {
                        if (updated.succeeded()) result.complete(); else result.fail(updated.cause());
                    });
                } catch (Exception ex) { result.fail(ex); }
            });
        } catch (Exception ex) { result.fail(ex); }
        return result;
    }

    public Future<Void> state(List<String> bindings, String state) {
        Future<Void> result = Future.succeededFuture();
        for (String binding : bindings) {
            result = result.compose(ignored -> {
                NativeAffinityOwner row = new NativeAffinityOwner();
                row.setBindingId(binding);
                row.setState(state);
                Future<Void> updated = Future.future();
                try {
                    queryExecutor.execute(row, Keys.UUID, QueryType.UPDATE).setHandler(handler -> {
                        if (handler.succeeded()) updated.complete(); else updated.fail(handler.cause());
                    });
                } catch (Exception ex) { updated.fail(ex); }
                return updated;
            });
        }
        return result;
    }
}

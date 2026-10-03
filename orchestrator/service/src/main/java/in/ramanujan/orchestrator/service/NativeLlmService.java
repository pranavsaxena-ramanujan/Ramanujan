package in.ramanujan.orchestrator.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import in.ramanujan.db.layer.schema.NativeAffinityOwner;
import in.ramanujan.orchestrator.base.NativeSessionKey;
import in.ramanujan.orchestrator.base.enums.Status;
import in.ramanujan.orchestrator.base.pojo.AsyncTask;
import in.ramanujan.orchestrator.data.dao.*;
import io.vertx.core.CompositeFuture;
import io.vertx.core.Future;
import io.vertx.core.Vertx;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

/** Native inference uses the existing durable async-task and host-assignment infrastructure. */
@Component
public class NativeLlmService {
    @Autowired private AsyncTaskDao asyncTaskDao;
    @Autowired private AsyncTaskHostMappingDao hostMappingDao;
    @Autowired private HostsDao hostsDao;
    @Autowired private NativeAffinityDao affinityDao;
    @Autowired private CapacityDao capacityDao;
    private final ObjectMapper mapper = new ObjectMapper();

    public static String binding(String cluster, String affinity, String session) {
        return NativeSessionKey.binding(cluster, affinity, session);
    }

    private static String session(String cluster, String session) {
        return NativeSessionKey.session(cluster, session);
    }

    private static String required(Map<String, Object> request, String field) {
        Object value = request.get(field);
        if (!(value instanceof String) || ((String) value).isEmpty() || ((String) value).length() > 200) {
            throw new IllegalArgumentException(field + " is required");
        }
        return (String) value;
    }

    private static String cluster(Map<String, Object> request) {
        Object value = request.get("clusterId");
        if (value != null && !(value instanceof String)) throw new IllegalArgumentException("clusterId must be a string");
        if (value != null && ((String) value).length() > 100) throw new IllegalArgumentException("clusterId is too long");
        return (String) value;
    }

    private static long timeout(Map<String, Object> request) {
        Object value = request.get("timeout");
        long seconds = value instanceof Number ? ((Number) value).longValue() : 1800L;
        if (seconds < 1 || seconds > 1800) throw new IllegalArgumentException("timeout must be between 1 and 1800 seconds");
        return seconds;
    }

    @SuppressWarnings("unchecked")
    private static List<String> files(Map<String, Object> stage) {
        if (stage.get("files") == null) return Collections.emptyList();
        if (!(stage.get("files") instanceof List)) throw new IllegalArgumentException("files must be an array");
        List<String> files = new ArrayList<>();
        for (Object file : (List<Object>) stage.get("files")) {
            if (!(file instanceof String)) throw new IllegalArgumentException("files must contain strings");
            files.add((String) file);
        }
        return files;
    }

    public Future<Map<String, Object>> execute(Map<String, Object> request, String op, Vertx vertx) {
        try {
            String cluster = cluster(request);
            long deadline = System.currentTimeMillis() + timeout(request) * 1000L;
            if ("chain".equals(op)) return chain(request, cluster, deadline, vertx);
            Map<String, Object> payload = new LinkedHashMap<>(request);
            payload.remove("clusterId");
            payload.remove("files");
            payload.put("op", op);
            payload.put("session", session(cluster, required(request, "session")));
            List<String> bindings = Collections.singletonList(binding(cluster,
                    required(request, "affinity"), required(request, "session")));
            return prepare(payload, bindings, files(request), cluster, deadline, vertx);
        } catch (Exception ex) {
            return Future.failedFuture(ex);
        }
    }

    @SuppressWarnings("unchecked")
    private Future<Map<String, Object>> chain(Map<String, Object> request, String cluster, long deadline, Vertx vertx) {
        Object stagesObject = request.get("stages");
        if (!(stagesObject instanceof List) || ((List<?>) stagesObject).isEmpty()) {
            return Future.failedFuture(new IllegalArgumentException("stages are required"));
        }
        List<Map<String, Object>> stages = (List<Map<String, Object>>) stagesObject;
        List<Future> owners = new ArrayList<>();
        for (Map<String, Object> stage : stages) {
            owners.add(affinityDao.get(binding(cluster, required(stage, "affinity"), required(stage, "session"))));
        }
        return CompositeFuture.all(owners).compose(ignored ->
                chainGroups(request, stages, owners, cluster, deadline, vertx, 0,
                        request.get("tokens"), request.get("hidden"), new ArrayList<>(), new ArrayList<>(), 0));
    }

    @SuppressWarnings("unchecked")
    private Future<Map<String, Object>> chainGroups(Map<String, Object> request, List<Map<String, Object>> stages,
            List<Future> owners, String cluster, long deadline, Vertx vertx, int start, Object tokens, Object hidden,
            List<Object> infos, List<Object> workers, int tasks) {
        NativeAffinityOwner first = (NativeAffinityOwner) owners.get(start).result();
        int end = start + 1;
        if (first != null) {
            while (end < stages.size()) {
                NativeAffinityOwner next = (NativeAffinityOwner) owners.get(end).result();
                if (next == null || !first.getHostId().equals(next.getHostId())) break;
                end++;
            }
        }
        List<String> bindings = new ArrayList<>();
        List<String> files = new ArrayList<>();
        List<Map<String, Object>> group = new ArrayList<>();
        for (Map<String, Object> stage : stages.subList(start, end)) {
            bindings.add(binding(cluster, required(stage, "affinity"), required(stage, "session")));
            files.addAll(files(stage));
            Map<String, Object> workerStage = new LinkedHashMap<>();
            workerStage.put("session", session(cluster, required(stage, "session")));
            if (stage.get("graph") != null) workerStage.put("graph", stage.get("graph"));
            group.add(workerStage);
        }
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("op", "chain");
        if (request.get("planId") != null) payload.put("planId", required(request, "planId"));
        payload.put("stages", group);
        payload.put("n", request.get("n"));
        payload.put("pos", request.get("pos"));
        if (tokens != null) payload.put("tokens", tokens);
        if (hidden != null) payload.put("hidden", hidden);
        if (end == stages.size() && request.get("output") != null) payload.put("output", request.get("output"));
        int nextStart = end;
        return prepare(payload, bindings, files, cluster, deadline, vertx).compose(result -> {
            if (result.get("error") != null) return Future.succeededFuture(result);
            Map<String, Object> data = result.get("data") instanceof Map
                    ? (Map<String, Object>) result.get("data") : Collections.emptyMap();
            if (data.get("infos") instanceof List) infos.addAll((List<Object>) data.get("infos"));
            workers.add(result.get("hostId"));
            if (nextStart == stages.size()) {
                Map<String, Object> response = new LinkedHashMap<>();
                if (data.containsKey("token")) response.put("token", data.get("token"));
                if (data.containsKey("output")) response.put("output", data.get("output"));
                response.put("infos", infos);
                response.put("workers", workers);
                response.put("tasks", tasks + 1);
                response.put("status", "SUCCESS");
                return Future.succeededFuture(response);
            }
            return chainGroups(request, stages, owners, cluster, deadline, vertx, nextStart,
                    null, data.get("output"), infos, workers, tasks + 1);
        });
    }

    @SuppressWarnings("unchecked")
    private Future<Map<String, Object>> prepare(Map<String, Object> payload, List<String> bindings,
            List<String> declaredFiles, String cluster, long deadline, Vertx vertx) {
        AsyncTask task = new AsyncTask();
        task.setUuid(UUID.randomUUID().toString());
        task.setClusterId(cluster);
        task.setStatus(Status.PROCESSING.getKeyName());
        task.setDebug(false);
        task.setLlm(payload);
        task.setNativeState("QUEUED");
        task.setNativeDeadline(deadline);
        task.setNativeBindings(bindings);
        Set<String> allowedFiles = new LinkedHashSet<>(declaredFiles);
        String position = NativeAffinityDao.position(payload);
        Future<Void> owners;
        if (payload.get("planId") != null) {
            if (capacityDao == null) return Future.failedFuture("Capacity admission is unavailable");
            owners = capacityDao.pin(cluster, required(payload, "planId"), bindings, payload, declaredFiles)
                    .map(host -> { task.setNativePreferredHost(host); return (Void) null; });
        } else owners = Future.succeededFuture();
        for (String binding : bindings) {
            owners = owners.compose(ignored -> affinityDao.get(binding).compose(owner -> {
                if (owner == null) return Future.succeededFuture();
                if (!Objects.equals(cluster, owner.getClusterId()) || !"ACTIVE".equals(owner.getState())
                        || owner.getLastUsed() == null || System.currentTimeMillis() - owner.getLastUsed() >= NativeAffinityDao.LEASE_MILLIS) {
                    return Future.failedFuture("Native session expired or failed; start a new session");
                }
                if (!"close".equals(position) && owner.getLastPosition() != null
                        && !"close".equals(owner.getLastPosition())
                        && Long.parseLong(position) <= Long.parseLong(owner.getLastPosition())) {
                    return Future.failedFuture("Stateful position was already submitted; do not replay it");
                }
                if (task.getNativePreferredHost() != null && !task.getNativePreferredHost().equals(owner.getHostId())) {
                    return Future.failedFuture("Native chain group has different owners");
                }
                task.setNativePreferredHost(owner.getHostId());
                try {
                    if (owner.getFiles() != null) allowedFiles.addAll(mapper.readValue(owner.getFiles(), List.class));
                } catch (Exception ex) { return Future.failedFuture(ex); }
                if (owner.getLastTask() != null) {
                    return asyncTaskDao.getAsyncTask(owner.getLastTask()).compose(previous ->
                            previous != null && "COMPLETED".equals(previous.getNativeState())
                                    ? Future.succeededFuture()
                                    : Future.failedFuture("Previous native task is unresolved; do not advance or replay the session"));
                }
                return Future.succeededFuture();
            }));
        }
        return owners.compose(ignored -> {
            task.setNativeFiles(new ArrayList<>(allowedFiles));
            return asyncTaskDao.insert(task).compose(inserted -> awaitTask(task, vertx));
        });
    }

    private Future<Map<String, Object>> awaitTask(AsyncTask submitted, Vertx vertx) {
        Future<Map<String, Object>> result = Future.future();
        AtomicBoolean checking = new AtomicBoolean();
        long timer = vertx.setPeriodic(100L, ignored -> {
            if (result.isComplete() || !checking.compareAndSet(false, true)) return;
            asyncTaskDao.getAsyncTask(submitted.getUuid()).compose(task -> {
                if (task == null || task.getUuid() == null) return Future.<Map<String, Object>>failedFuture("Native task disappeared");
                if (task.getNativeResult() != null) return Future.succeededFuture(task.getNativeResult());
                if (System.currentTimeMillis() >= submitted.getNativeDeadline()) {
                    return fail(task, "Native task timed out; start a new session instead of retrying stateful work", true)
                            .map(failed -> failed);
                }
                if ("QUEUED".equals(task.getNativeState())) {
                    // The capacity-plan host is pinned in memory only; the reloaded row does not carry it.
                    if (task.getNativePreferredHost() == null) task.setNativePreferredHost(submitted.getNativePreferredHost());
                    return refreshPlacement(task).compose(ignoredPlacement -> hostsDao.getMachine(task, false))
                            .map(host -> (Map<String, Object>) null);
                }
                return Future.succeededFuture((Map<String, Object>) null);
            }).setHandler(checked -> {
                checking.set(false);
                if (checked.failed()) {
                    fail(submitted, checked.cause().getMessage(), false).setHandler(failed -> {
                        if (failed.succeeded()) result.tryComplete(failed.result()); else result.tryFail(checked.cause());
                    });
                }
                else if (checked.result() != null) result.tryComplete(checked.result());
            });
        });
        return result.map(payload -> {
            vertx.cancelTimer(timer);
            return payload;
        }).recover(error -> {
            vertx.cancelTimer(timer);
            return Future.failedFuture(error);
        });
    }

    private Future<Void> refreshPlacement(AsyncTask task) {
        Future<Void> result = Future.succeededFuture();
        for (String binding : task.getNativeBindings()) {
            result = result.compose(ignored -> affinityDao.get(binding).compose(owner -> {
                if (owner == null) return Future.succeededFuture();
                if (!"ACTIVE".equals(owner.getState()) || owner.getLastUsed() == null
                        || System.currentTimeMillis() - owner.getLastUsed() >= NativeAffinityDao.LEASE_MILLIS) {
                    return Future.failedFuture("Native session is unavailable; start a new session");
                }
                String position = NativeAffinityDao.position(task.getLlm());
                if (!task.getUuid().equals(owner.getLastTask()) && !"close".equals(position)
                        && owner.getLastPosition() != null && !"close".equals(owner.getLastPosition())
                        && Long.parseLong(position) <= Long.parseLong(owner.getLastPosition())) {
                    return Future.failedFuture("Stateful position was already submitted; do not replay it");
                }
                if (task.getNativePreferredHost() != null && !task.getNativePreferredHost().equals(owner.getHostId())) {
                    return Future.failedFuture("Native session owner changed");
                }
                task.setNativePreferredHost(owner.getHostId());
                return Future.succeededFuture();
            }));
        }
        return result;
    }

    public Future<Void> complete(AsyncTask task, Map<String, Object> payload) {
        if (!"ASSIGNED".equals(task.getNativeState()) || task.getNativeDeadline() == null
                || System.currentTimeMillis() >= task.getNativeDeadline()) {
            return Future.failedFuture(new SecurityException("Native task is not assigned or has expired"));
        }
        boolean failed = payload.get("error") != null;
        Map<String, Object> update = new LinkedHashMap<>();
        update.put("nativeResult", payload);
        update.put("nativeState", failed ? "FAILED" : "COMPLETED");
        update.put("status", failed ? Status.FAILURE.getKeyName() : Status.SUCCESS.getKeyName());
        // Record session/plan state before publishing the result: the submitter returns as soon as
        // nativeResult is visible and may release the capacity plan immediately after a close.
        Future<Void> recorded;
        if (failed) {
            recorded = affinityDao.state(task.getNativeBindings(), "FAILED");
        } else {
            boolean closed = "close".equals(task.getLlm().get("op"));
            recorded = closed ? affinityDao.state(task.getNativeBindings(), "CLOSED") : Future.succeededFuture();
            if (task.getLlm().get("planId") != null) {
                recorded = recorded.compose(done -> capacityDao.completed(task.getClusterId(),
                        (String) task.getLlm().get("planId"), task.getNativeBindings(), closed));
            }
        }
        return recorded.map(done -> (Throwable) null).recover(Future::succeededFuture).compose(stateError ->
                asyncTaskDao.update(task.getUuid(), update).compose(ignored ->
                        hostMappingDao.removeMapping(task.getHostAssigned(), task.getUuid())).compose(ignored ->
                        stateError == null ? Future.<Void>succeededFuture() : Future.<Void>failedFuture(stateError)));
    }

    public Future<Map<String, Object>> fail(AsyncTask task, String error, boolean timedOut) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("error", error);
        if (timedOut) payload.put("timedOut", true);
        Map<String, Object> update = new LinkedHashMap<>();
        update.put("nativeResult", payload);
        update.put("nativeState", "FAILED");
        update.put("status", Status.FAILURE.getKeyName());
        return asyncTaskDao.update(task.getUuid(), update).compose(ignored ->
                hostMappingDao.deleteTask(task.getUuid())).compose(ignored ->
                failOwnedBindings(task)).map(ignored -> payload);
    }

    private Future<Void> failOwnedBindings(AsyncTask task) {
        Future<Void> result = Future.succeededFuture();
        for (String binding : task.getNativeBindings()) {
            result = result.compose(ignored -> affinityDao.get(binding).compose(owner ->
                    owner != null && task.getUuid().equals(owner.getLastTask())
                            ? affinityDao.state(Collections.singletonList(binding), "FAILED")
                            : Future.succeededFuture()));
        }
        return result;
    }
}

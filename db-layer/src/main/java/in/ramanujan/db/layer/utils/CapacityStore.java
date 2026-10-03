package in.ramanujan.db.layer.utils;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.vertx.core.Future;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.sql.*;
import java.util.*;

/** Serializes admission within a room, including across orchestrator processes. */
@Component
public class CapacityStore {
    @Autowired private QueryExecutor queryExecutor;
    private final ObjectMapper mapper = new ObjectMapper();
    private final Map<String, State> memory = new HashMap<>();

    public static class State {
        public Map<String, Map<String, Object>> hosts = new LinkedHashMap<>();
        public Map<String, Map<String, Object>> plans = new LinkedHashMap<>();
    }

    @FunctionalInterface
    public interface Work<T> {
        T apply(State state) throws Exception;
    }

    public <T> Future<T> atomic(String cluster, Work<T> work) {
        if (queryExecutor.isInMemory()) {
            synchronized (memory) {
                try {
                    State original = memory.getOrDefault(cluster, new State());
                    State copy = mapper.readValue(mapper.writeValueAsBytes(original), State.class);
                    prune(copy);
                    T value = work.apply(copy);
                    memory.put(cluster, copy);
                    return Future.succeededFuture(value);
                } catch (Exception error) { return Future.failedFuture(error); }
            }
        }
        return locked(cluster, work, 3);
    }

    private static boolean deadlock(Throwable error) {
        for (Throwable cause = error; cause != null; cause = cause.getCause()) {
            if (cause instanceof SQLException && "40001".equals(((SQLException) cause).getSQLState())) return true;
        }
        return false;
    }

    private static boolean lockRoom(Connection connection, String cluster) throws SQLException {
        try (PreparedStatement lock = connection.prepareStatement(
                "SELECT clusterId FROM capacityClusterLock WHERE clusterId=? FOR UPDATE")) {
            lock.setString(1, cluster);
            try (ResultSet row = lock.executeQuery()) { return row.next(); }
        }
    }

    private <T> Future<T> locked(String cluster, Work<T> work, int attempts) {
        Future<T> result = Future.future();
        queryExecutor.<T>transaction(connection -> {
            // Lock an existing row first: INSERT IGNORE on a present key takes a shared lock, and two
            // such holders upgrading to FOR UPDATE deadlock (concurrent capacity pings and plans).
            if (!lockRoom(connection, cluster)) {
                try (PreparedStatement insert = connection.prepareStatement(
                        "INSERT IGNORE INTO capacityClusterLock (clusterId) VALUES (?)")) {
                    insert.setString(1, cluster);
                    insert.executeUpdate();
                }
                if (!lockRoom(connection, cluster)) throw new SQLException("Capacity room lock missing");
            }
            State state = new State();
            Map<String, String> hostRows = load(connection, "workerCapacity", "hostId", "snapshot", cluster, state.hosts);
            Map<String, String> planRows = load(connection, "llmCapacityPlan", "planId", "plan", cluster, state.plans);
            prune(state);
            T value = work.apply(state);
            save(connection, "workerCapacity", "hostId", "snapshot", cluster, state.hosts, hostRows);
            save(connection, "llmCapacityPlan", "planId", "plan", cluster, state.plans, planRows);
            return value;
        }).setHandler(done -> {
            if (done.succeeded()) result.complete(done.result());
            else if (attempts > 1 && deadlock(done.cause())) locked(cluster, work, attempts - 1).setHandler(result);
            else result.fail(done.cause());
        });
        return result;
    }

    private void prune(State state) {
        long cutoff = System.currentTimeMillis() - 24 * 60 * 60_000L;
        state.plans.values().removeIf(plan -> "RELEASED".equals(plan.get("state"))
                && plan.get("releasedAt") instanceof Number && ((Number) plan.get("releasedAt")).longValue() < cutoff);
    }

    @SuppressWarnings("unchecked")
    private Map<String, String> load(Connection connection, String table, String key, String column,
                                    String cluster, Map<String, Map<String, Object>> target) throws Exception {
        Map<String, String> originals = new HashMap<>();
        try (PreparedStatement select = connection.prepareStatement(
                "SELECT " + key + "," + column + " FROM " + table + " WHERE clusterId=?")) {
            select.setString(1, cluster);
            try (ResultSet rows = select.executeQuery()) {
                while (rows.next()) {
                    String json = rows.getString(2);
                    originals.put(rows.getString(1), json);
                    target.put(rows.getString(1), mapper.readValue(json, Map.class));
                }
            }
        }
        return originals;
    }

    private void save(Connection connection, String table, String key, String column, String cluster,
                      Map<String, Map<String, Object>> values, Map<String, String> originals) throws Exception {
        for (String deleted : originals.keySet()) if (!values.containsKey(deleted)) {
            try (PreparedStatement remove = connection.prepareStatement(
                    "DELETE FROM " + table + " WHERE " + key + "=? AND clusterId=?")) {
                remove.setString(1, deleted);
                remove.setString(2, cluster);
                remove.executeUpdate();
            }
        }
        for (Map.Entry<String, Map<String, Object>> entry : values.entrySet()) {
            String json = mapper.writeValueAsString(entry.getValue());
            if (json.equals(originals.get(entry.getKey()))) continue;
            if (originals.containsKey(entry.getKey())) {
                try (PreparedStatement update = connection.prepareStatement(
                        "UPDATE " + table + " SET " + column + "=?,updatedAt=? WHERE " + key + "=? AND clusterId=?")) {
                    update.setString(1, json);
                    update.setLong(2, System.currentTimeMillis());
                    update.setString(3, entry.getKey());
                    update.setString(4, cluster);
                    if (update.executeUpdate() != 1) throw new SQLException("Capacity row changed room");
                }
            } else {
                try (PreparedStatement insert = connection.prepareStatement(
                        "INSERT INTO " + table + " (" + key + ",clusterId," + column + ",updatedAt) VALUES (?,?,?,?)")) {
                    insert.setString(1, entry.getKey());
                    insert.setString(2, cluster);
                    insert.setString(3, json);
                    insert.setLong(4, System.currentTimeMillis());
                    insert.executeUpdate();
                }
            }
        }
    }
}

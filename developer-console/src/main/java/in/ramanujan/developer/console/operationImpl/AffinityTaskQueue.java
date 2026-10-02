package in.ramanujan.developer.console.operationImpl;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.LongSupplier;

/**
 * Homelab task queue with optional sticky affinity (for example, a model shard id).
 *
 * Tasks without affinity go to any polling worker, as before. The first task for an
 * affinity binds it to the polling worker when that worker is under its affinity limit
 * and no other live worker owns fewer affinities; later tasks for that affinity go only
 * to the owner, so each worker caches and executes just its own shards. An idle owner
 * that has not polled for {@link #STALE_MILLIS} (or a busy one for
 * {@link #BUSY_STALE_MILLIS}) loses its affinities to the next eligible worker.
 */
final class AffinityTaskQueue<T> {
    static final long LIVE_MILLIS = 10_000L;
    static final long STALE_MILLIS = 60_000L;
    // A single-threaded worker does not poll while it executes, so a busy owner stays
    // live much longer; the bound still frees affinities of a worker that died mid-task.
    static final long BUSY_STALE_MILLIS = 30 * 60_000L;

    private static final class Entry<T> {
        final T task;
        final String affinity;
        final String clusterId;

        Entry(T task, String affinity, String clusterId) {
            this.task = task;
            this.affinity = scoped(clusterId, affinity);
            this.clusterId = clusterId;
        }
    }

    private final List<Entry<T>> pending = new ArrayList<>();
    private final Map<String, String> owners = new LinkedHashMap<>();
    private final Map<String, Long> lastSeen = new HashMap<>();
    private final Map<String, Integer> limits = new HashMap<>();
    private final Map<String, Integer> busy = new HashMap<>();
    private final Map<String, String> hostClusters = new HashMap<>();
    private final Map<String, String> hostIds = new HashMap<>();
    private final LongSupplier clock;

    AffinityTaskQueue() {
        this(System::currentTimeMillis);
    }

    AffinityTaskQueue(LongSupplier clock) {
        this.clock = clock;
    }

    synchronized void add(T task, String affinity) {
        add(task, affinity, null);
    }

    synchronized void add(T task, String affinity, String clusterId) {
        pending.add(new Entry<>(task, affinity, clusterId));
        notifyAll();
    }

    synchronized void remove(T task) {
        pending.removeIf(entry -> entry.task == task);
        notifyAll();
    }

    private static String scoped(String clusterId, String value) {
        if (value == null) return null;
        return (clusterId == null ? "-1:" : clusterId.length() + ":" + clusterId + ":") + value;
    }

    synchronized int size() {
        return pending.size();
    }

    synchronized Map<String, String> owners() {
        return owners(null);
    }

    synchronized Map<String, String> owners(String clusterId) {
        String prefix = clusterId == null ? "-1:" : clusterId.length() + ":" + clusterId + ":";
        Map<String, String> result = new LinkedHashMap<>();
        owners.forEach((key, value) -> {
            if (key.startsWith(prefix)) result.put(key.substring(prefix.length()), hostIds.get(value));
        });
        return result;
    }

    /** Returns the first eligible task for {@code host}, waiting up to {@code timeoutMillis}. */
    synchronized T poll(String host, int affinityLimit, long timeoutMillis) throws InterruptedException {
        return poll(host, affinityLimit, timeoutMillis, null);
    }

    synchronized T poll(String host, int affinityLimit, long timeoutMillis, String clusterId) throws InterruptedException {
        String hostId = host;
        host = scoped(clusterId, host);
        long deadline = System.currentTimeMillis() + timeoutMillis;
        while (true) {
            long now = clock.getAsLong();
            if (host != null) {
                lastSeen.put(host, now);
                limits.put(host, Math.max(0, affinityLimit));
                hostClusters.put(host, clusterId);
                hostIds.put(host, hostId);
            }
            for (Iterator<Entry<T>> it = pending.iterator(); it.hasNext(); ) {
                Entry<T> entry = it.next();
                if ((entry.clusterId == null || entry.clusterId.equals(clusterId))
                        && eligible(entry.affinity, host, now, entry.clusterId)) {
                    it.remove();
                    if (entry.affinity != null) {
                        owners.put(entry.affinity, host);
                    }
                    if (host != null) {
                        busy.merge(host, 1, Integer::sum);
                    }
                    return entry.task;
                }
            }
            long remaining = deadline - System.currentTimeMillis();
            if (remaining <= 0) {
                return null;
            }
            wait(remaining);
        }
    }

    /** Records that {@code host} finished a task handed out by {@link #poll}. */
    synchronized void completed(String host) {
        completed(host, null);
    }

    synchronized void completed(String host, String clusterId) {
        host = scoped(clusterId, host);
        if (host == null) return;
        lastSeen.put(host, clock.getAsLong());
        Integer count = busy.get(host);
        if (count != null) {
            if (count <= 1) busy.remove(host); else busy.put(host, count - 1);
        }
        notifyAll();
    }

    private boolean eligible(String affinity, String host, long now, String clusterId) {
        if (affinity == null) return true;
        if (host == null) return false;
        String owner = owners.get(affinity);
        if (host.equals(owner)) return true;
        if (owner != null && !isAlive(owner, now, STALE_MILLIS)) {
            owners.remove(affinity);
            owner = null;
        }
        if (owner != null) return false;
        int mine = owned(host);
        if (mine >= limits.getOrDefault(host, Integer.MAX_VALUE)) return false;
        for (String other : lastSeen.keySet()) {
            if ((clusterId == null || clusterId.equals(hostClusters.get(other)))
                    && !other.equals(host) && isAlive(other, now, LIVE_MILLIS)
                    && owned(other) < mine && owned(other) < limits.getOrDefault(other, Integer.MAX_VALUE)) {
                return false;
            }
        }
        return true;
    }

    private boolean isAlive(String host, long now, long window) {
        long age = now - lastSeen.getOrDefault(host, Long.MIN_VALUE / 2);
        return age < (busy.containsKey(host) ? Math.max(window, BUSY_STALE_MILLIS) : window);
    }

    private int owned(String host) {
        int count = 0;
        for (String owner : owners.values()) {
            if (owner.equals(host)) count++;
        }
        return count;
    }
}

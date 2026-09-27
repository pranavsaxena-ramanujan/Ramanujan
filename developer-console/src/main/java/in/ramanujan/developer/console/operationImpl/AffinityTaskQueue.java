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

        Entry(T task, String affinity) {
            this.task = task;
            this.affinity = affinity;
        }
    }

    private final List<Entry<T>> pending = new ArrayList<>();
    private final Map<String, String> owners = new LinkedHashMap<>();
    private final Map<String, Long> lastSeen = new HashMap<>();
    private final Map<String, Integer> limits = new HashMap<>();
    private final Map<String, Integer> busy = new HashMap<>();
    private final LongSupplier clock;

    AffinityTaskQueue() {
        this(System::currentTimeMillis);
    }

    AffinityTaskQueue(LongSupplier clock) {
        this.clock = clock;
    }

    synchronized void add(T task, String affinity) {
        pending.add(new Entry<>(task, affinity));
        notifyAll();
    }

    synchronized int size() {
        return pending.size();
    }

    synchronized Map<String, String> owners() {
        return new LinkedHashMap<>(owners);
    }

    /** Returns the first eligible task for {@code host}, waiting up to {@code timeoutMillis}. */
    synchronized T poll(String host, int affinityLimit, long timeoutMillis) throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMillis;
        while (true) {
            long now = clock.getAsLong();
            if (host != null) {
                lastSeen.put(host, now);
                limits.put(host, Math.max(0, affinityLimit));
            }
            for (Iterator<Entry<T>> it = pending.iterator(); it.hasNext(); ) {
                Entry<T> entry = it.next();
                if (eligible(entry.affinity, host, now)) {
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
        if (host == null) return;
        lastSeen.put(host, clock.getAsLong());
        Integer count = busy.get(host);
        if (count != null) {
            if (count <= 1) busy.remove(host); else busy.put(host, count - 1);
        }
        notifyAll();
    }

    private boolean eligible(String affinity, String host, long now) {
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
            if (!other.equals(host) && isAlive(other, now, LIVE_MILLIS)
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

package in.ramanujan.developer.console.operationImpl;

import org.junit.Test;

import java.util.concurrent.atomic.AtomicLong;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

public class AffinityTaskQueueTest {
    private final AtomicLong now = new AtomicLong(1_000_000L);
    private final AffinityTaskQueue<String> queue = new AffinityTaskQueue<>(now::get);

    private String poll(String host, int limit) throws InterruptedException {
        return queue.poll(host, limit, 0);
    }

    @Test
    public void taggedTasksAndAffinityOwnersAreClusterIsolated() throws Exception {
        queue.add("a", "same", "A");
        assertNull(queue.poll("worker", 1, 0, "B"));
        assertNull(queue.poll("global", 1, 0));
        assertEquals("a", queue.poll("worker", 1, 0, "A"));
        queue.completed("worker", "A");
        queue.add("b", "same", "B");
        assertEquals("b", queue.poll("worker", 1, 0, "B"));
        assertEquals("worker", queue.owners("A").get("same"));
        assertEquals("worker", queue.owners("B").get("same"));
        assertNull(queue.owners().get("same"));
    }

    @Test
    public void untaggedTasksRunInEveryClusterAndStaleTaggedWorkStaysInCluster() throws Exception {
        for (String cluster : new String[]{null, "A", "B"}) {
            queue.add("free", null);
            assertEquals("free", queue.poll("worker", 2, 0, cluster));
            queue.completed("worker", cluster);
        }
        queue.add("first", "affinity", "A");
        assertEquals("first", queue.poll("worker", 2, 0, "A"));
        queue.completed("worker", "A");
        queue.add("retry", "affinity", "A");
        now.addAndGet(AffinityTaskQueue.STALE_MILLIS + 1);
        assertNull(queue.poll("worker", 2, 0, "B"));
        assertEquals("retry", queue.poll("replacement", 2, 0, "A"));
    }

    @Test
    public void spreadsAffinitiesAcrossLiveWorkersAndKeepsThemSticky() throws Exception {
        assertNull(poll("a", 2));
        assertNull(poll("b", 2));

        queue.add("s0-step0", "s0");
        assertEquals("s0-step0", poll("a", 2));
        queue.completed("a");

        queue.add("s1-step0", "s1");
        assertNull("a already owns more shards than live worker b", poll("a", 2));
        assertEquals("s1-step0", poll("b", 2));
        queue.completed("b");

        queue.add("s0-step1", "s0");
        assertNull("s0 belongs to a", poll("b", 2));
        assertEquals("s0-step1", poll("a", 2));
        queue.completed("a");
        assertEquals("a", queue.owners().get("s0"));
        assertEquals("b", queue.owners().get("s1"));
    }

    @Test
    public void respectsPerWorkerAffinityLimit() throws Exception {
        queue.add("s0", "s0");
        assertEquals("s0", poll("a", 1));
        queue.completed("a");
        queue.add("s1", "s1");
        assertNull(poll("a", 1));
        assertEquals(1, queue.size());
        assertEquals("s1", poll("b", 1));
    }

    @Test
    public void staleOwnerLosesAffinity() throws Exception {
        queue.add("s0", "s0");
        assertEquals("s0", poll("a", 4));
        queue.completed("a");

        queue.add("again", "s0");
        now.addAndGet(AffinityTaskQueue.STALE_MILLIS - 1);
        assertNull(poll("b", 4));
        now.addAndGet(2);
        assertEquals("again", poll("b", 4));
        assertEquals("b", queue.owners().get("s0"));
    }

    @Test
    public void busyOwnerStaysLiveUntilBusyBound() throws Exception {
        queue.add("long", "s0");
        assertEquals("long", poll("a", 4));
        queue.add("next", "s0");
        now.addAndGet(AffinityTaskQueue.STALE_MILLIS * 2);
        assertNull("a is still executing", poll("b", 4));
        now.addAndGet(AffinityTaskQueue.BUSY_STALE_MILLIS);
        assertEquals("next", poll("b", 4));
    }

    @Test
    public void tasksWithoutAffinityGoToAnyWorkerInOrder() throws Exception {
        queue.add("s0", "s0");
        assertEquals("s0", poll("a", 1));
        queue.add("pinned", "s0");
        queue.add("free", null);
        assertEquals("pinned task is skipped, not blocking", "free", poll("b", 1));
        assertEquals("free tasks also serve anonymous pollers", null, queue.poll(null, 1, 0));
        queue.add("anon", null);
        assertEquals("anon", queue.poll(null, 1, 0));
    }

    @Test
    public void pollWaitsForLaterTask() throws Exception {
        Thread producer = new Thread(() -> {
            try {
                Thread.sleep(50);
            } catch (InterruptedException ignored) {
            }
            queue.add("late", null);
        });
        producer.start();
        assertEquals("late", queue.poll("a", 1, 2_000));
        producer.join();
    }
}

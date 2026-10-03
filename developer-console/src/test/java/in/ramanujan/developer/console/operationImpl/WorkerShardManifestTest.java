package in.ramanujan.developer.console.operationImpl;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.nio.file.Path;
import java.util.*;

import static org.junit.Assert.*;

public class WorkerShardManifestTest {
    @Rule public TemporaryFolder folder = new TemporaryFolder();

    private static Map<String, Object> map(Object... pairs) {
        Map<String, Object> result = new LinkedHashMap<>();
        for (int i = 0; i < pairs.length; i += 2) result.put((String) pairs[i], pairs[i + 1]);
        return result;
    }

    private static Map<String, Object> hyper() {
        return map("architecture", "qwen35", "ssm", map("conv", 4, "d_state", 128), "eps", 1.0E-6,
                "dim", 5120, "rope_theta", 1.0E7);
    }

    private static Map<String, Object> graph(boolean embed, int first, int last, boolean head) {
        Map<String, Object> graph = map("hyper", hyper(), "weights", "resident", "layers", new ArrayList<>());
        if (embed) graph.put("embed", map("file", "/m/token_embd.bin"));
        if (first <= last) graph.put("rope_freqs", map("file", "/m/rope.bin"));
        for (int index = first; index <= last; index++) {
            ((List<Object>) graph.get("layers")).add(map("index", index, "tensors",
                    map("attn", map("file", "/m/blk." + index + ".attn.bin"), "ffn", map("file", "/m/blk." + index + ".ffn.bin"))));
        }
        if (head) graph.put("head", map("norm", map("file", "/m/norm.bin"), "output", map("file", "/m/output.bin")));
        return graph;
    }

    private static Map<String, Long> cached(String... paths) {
        Map<String, Long> result = new HashMap<>();
        for (String path : paths) result.put(path, 10L);
        return result;
    }

    @Test
    public void modelKeyMatchesTheOrchestrator() {
        // Same literal as CapacityDaoTest.MODEL_KEY: both sides must name a model identically.
        assertEquals("878d4985f9a5823f297ba33895d9689943409f70b93d4f5492e1fb8ce2cb80e2", WorkerShardManifest.modelKey(hyper()));
    }

    @Test
    @SuppressWarnings("unchecked")
    public void reportsCachedOffsetsAcrossRestartsAndDropsEvictedUnits() throws Exception {
        Path root = folder.getRoot().toPath();
        WorkerShardManifest manifest = new WorkerShardManifest(root, "http://room/worker/a");
        manifest.record(graph(true, 0, 2, false));
        manifest.record(graph(false, 3, 4, true));
        Map<String, Long> all = cached("/m/token_embd.bin", "/m/rope.bin", "/m/norm.bin", "/m/output.bin");
        for (int index = 0; index <= 4; index++) all.putAll(cached("/m/blk." + index + ".attn.bin", "/m/blk." + index + ".ffn.bin"));

        // A fresh instance (worker restart) reads the same manifest from the cache directory.
        List<Map<String, Object>> shards = new WorkerShardManifest(root, "http://room/worker/a").report(all);
        assertEquals(1, shards.size());
        Map<String, Object> shard = shards.get(0);
        assertEquals(WorkerShardManifest.modelKey(hyper()), shard.get("model"));
        assertEquals(true, shard.get("embed"));
        assertEquals(true, shard.get("head"));
        assertEquals(Collections.singletonList(Arrays.asList(0, 5)), shard.get("layers"));
        assertEquals(140L, shard.get("bytes"));

        // An evicted layer splits the range; a missing head file drops the head.
        all.remove("/m/blk.2.ffn.bin");
        all.remove("/m/output.bin");
        shard = manifest.report(all).get(0);
        assertEquals(Arrays.asList(Arrays.asList(0, 2), Arrays.asList(3, 5)), shard.get("layers"));
        assertEquals(false, shard.get("head"));

        // Another room's gateway never sees these weights.
        assertTrue(new WorkerShardManifest(root, "http://other/worker/b").report(all).isEmpty());
    }
}

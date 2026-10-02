package in.ramanujan.developer.console.operationImpl;

import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class LlmChainGroupingTest {

    private static List<Map<String, Object>> stages(String... affinities) {
        List<Map<String, Object>> stages = new ArrayList<>();
        for (String affinity : affinities) {
            Map<String, Object> stage = new HashMap<>();
            stage.put("affinity", affinity);
            stages.add(stage);
        }
        return stages;
    }

    private static List<Integer> groupEnds(List<Map<String, Object>> stages, Map<String, String> owners) {
        List<Integer> ends = new ArrayList<>();
        for (int i = 0; i < stages.size(); i = ends.get(ends.size() - 1)) {
            ends.add(ExecuteInlineHomelabServer.groupEnd(stages, i, owners));
        }
        return ends;
    }

    @Test
    public void consecutiveStagesOfOneWorkerFormOneGroup() {
        Map<String, String> owners = new HashMap<>();
        owners.put("s0", "A");
        owners.put("s1", "A");
        owners.put("s2", "B");
        owners.put("s3", "A");
        assertEquals(java.util.Arrays.asList(2, 3, 5), groupEnds(stages("s0", "s1", "s2", "s3", "s0"), owners));
    }

    @Test
    public void stagesWithoutAnOwnerGoOutAlone() {
        Map<String, String> owners = new HashMap<>();
        owners.put("s0", "A");
        assertEquals(java.util.Arrays.asList(1, 2, 3), groupEnds(stages("s0", "s1", "s2"), owners));
        assertEquals(java.util.Arrays.asList(1, 2), groupEnds(stages("s1", "s2"), new HashMap<String, String>()));
    }
}

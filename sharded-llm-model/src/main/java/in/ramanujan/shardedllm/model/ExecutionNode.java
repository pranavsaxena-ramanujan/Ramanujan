package in.ramanujan.shardedllm.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class ExecutionNode {
    private String nodeId;
    private String shardId;
    private String entrypoint;
    private List<String> dependsOn = new ArrayList<>();
    private Map<String, String> inputBindings = new LinkedHashMap<>();
    private Map<String, String> outputBindings = new LinkedHashMap<>();
}
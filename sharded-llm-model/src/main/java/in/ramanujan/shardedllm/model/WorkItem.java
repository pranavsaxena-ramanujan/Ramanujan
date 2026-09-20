package in.ramanujan.shardedllm.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class WorkItem {
    private String protocolVersion;
    private String workId;
    private String sessionId;
    private String command;
    private String shardId;
    private String graphNodeId;
    private String entrypoint;
    private long checkpointVersion;
    private long deadlineEpochMillis;
    private List<TensorReference> inputs = new ArrayList<>();
    private Map<String, String> attributes = new LinkedHashMap<>();
}
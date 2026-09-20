package in.ramanujan.shardedllm.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class SessionStatus {
    private String sessionId;
    private String packageId;
    private String executionMode;
    private String status;
    private long checkpointVersion;
    private List<TensorReference> outputs = new ArrayList<>();
    private Map<String, String> metrics = new LinkedHashMap<>();
    private String error;
}
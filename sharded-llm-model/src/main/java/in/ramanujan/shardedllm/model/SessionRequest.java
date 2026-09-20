package in.ramanujan.shardedllm.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class SessionRequest {
    private String protocolVersion;
    private String packageId;
    private String executionMode;
    private String entrypoint;
    private List<TensorReference> inputs = new ArrayList<>();
    private Map<String, String> options = new LinkedHashMap<>();
}
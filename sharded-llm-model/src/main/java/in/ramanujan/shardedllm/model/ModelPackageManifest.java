package in.ramanujan.shardedllm.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelPackageManifest {
    private String schemaVersion;
    private String packageId;
    private String architectureId;
    private String architectureVersion;
    private String sourceFormat;
    private List<String> capabilities = new ArrayList<>();
    private List<ExecutionShardManifest> shards = new ArrayList<>();
    private List<ExecutionNode> executionGraph = new ArrayList<>();
    private Map<String, String> tokenizer = new LinkedHashMap<>();
    private Map<String, String> metadata = new LinkedHashMap<>();
}
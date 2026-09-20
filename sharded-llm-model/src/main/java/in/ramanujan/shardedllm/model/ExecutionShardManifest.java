package in.ramanujan.shardedllm.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class ExecutionShardManifest {
    private String shardId;
    private String irPath;
    private long diskBytes;
    private long estimatedResidentBytes;
    private List<String> capabilities = new ArrayList<>();
    private List<ExecutionEntrypoint> entrypoints = new ArrayList<>();
    private List<StateContract> states = new ArrayList<>();
    private Map<String, String> checksums = new LinkedHashMap<>();
    private Map<String, String> adapterMetadata = new LinkedHashMap<>();
}
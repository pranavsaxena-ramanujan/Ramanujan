package in.ramanujan.shardedllm.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class DeviceRegistration {
    private String protocolVersion;
    private String deviceId;
    private String workerUrl;
    private long diskCapacityBytes;
    private long residentBudgetBytes;
    private List<String> capabilities = new ArrayList<>();
    private List<String> cachedPackageIds = new ArrayList<>();
    private List<String> assignedShardIds = new ArrayList<>();
    private Map<String, String> accelerator = new LinkedHashMap<>();
}
package in.ramanujan.shardedllm.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class ExecutionEntrypoint {
    private String name;
    private String commandId;
    private String capability;
    private List<TensorContract> inputs = new ArrayList<>();
    private List<TensorContract> outputs = new ArrayList<>();
    private List<String> stateNames = new ArrayList<>();
}
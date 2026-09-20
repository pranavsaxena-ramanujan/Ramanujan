package in.ramanujan.shardedllm.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class StateContract {
    private TensorContract tensor;
    private String owner;
    private String persistence;
    private boolean mutable;
    private boolean checkpointed;
}
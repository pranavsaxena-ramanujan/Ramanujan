package in.ramanujan.shardedllm.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class TensorReference {
    private String name;
    private String dataType;
    private List<String> shape = new ArrayList<>();
    private long byteLength;
    private String contentLocation;
    private String checksum;
}
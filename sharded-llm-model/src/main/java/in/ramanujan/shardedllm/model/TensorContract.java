package in.ramanujan.shardedllm.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class TensorContract {
    private String name;
    private String dataType;
    private List<String> shape = new ArrayList<>();
    private String layout;
    private String role;
    private Map<String, String> attributes = new LinkedHashMap<>();
}
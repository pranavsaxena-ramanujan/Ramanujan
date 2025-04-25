package in.ramanujan.pojo.ruleEngineInputUnitsExt.array;

import lombok.Data;
import java.util.List;

@Data
public class RedefineArrayCommand {
    private String id;
    private String arrayId;
    private List<Integer> newDimensions;
    private Object initialValue; // Optional: can be null or a map of index->value
}

package in.ramanujan.pojo.ruleEngineInputUnitsExt.array;

import lombok.Data;
import java.util.List;

@Data
public class RedefineArrayCommand {
    private String id;
    private String arrayId;
    private List<String> newDimensions; 
}

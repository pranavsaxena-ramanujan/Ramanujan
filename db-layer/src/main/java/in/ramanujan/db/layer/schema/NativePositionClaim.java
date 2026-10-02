package in.ramanujan.db.layer.schema;

import in.ramanujan.db.layer.annotations.ColumnName;
import in.ramanujan.db.layer.annotations.PrimaryKey;
import in.ramanujan.db.layer.annotations.Table;
import in.ramanujan.db.layer.constants.Keys;
import lombok.Data;

@Data
@Table("nativePositionClaim")
public class NativePositionClaim {
    @ColumnName("claimId")
    @PrimaryKey(keyValue = Keys.UUID, order = "1")
    public String claimId;
    @ColumnName("taskUuid")
    public String taskUuid;
    @ColumnName("bindingId")
    public String bindingId;
    @ColumnName("position")
    public String position;
}

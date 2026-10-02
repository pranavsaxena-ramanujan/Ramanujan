package in.ramanujan.db.layer.schema;

import in.ramanujan.db.layer.annotations.ColumnName;
import in.ramanujan.db.layer.annotations.PrimaryKey;
import in.ramanujan.db.layer.annotations.Table;
import in.ramanujan.db.layer.constants.Keys;
import lombok.Data;

@Data
@Table("nativeTaskDelivery")
public class NativeTaskDelivery {
    @ColumnName("uuid")
    @PrimaryKey(keyValue = Keys.UUID, order = "1")
    public String uuid;
    @ColumnName("taskUuid")
    public String taskUuid;
    @ColumnName("hostId")
    public String hostId;
    @ColumnName("clusterId")
    public String clusterId;
    @ColumnName("deliveredAt")
    public Long deliveredAt;
}

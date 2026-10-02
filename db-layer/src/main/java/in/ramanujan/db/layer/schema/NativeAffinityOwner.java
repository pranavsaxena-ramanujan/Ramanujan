package in.ramanujan.db.layer.schema;

import in.ramanujan.db.layer.annotations.ColumnName;
import in.ramanujan.db.layer.annotations.PrimaryKey;
import in.ramanujan.db.layer.annotations.Table;
import in.ramanujan.db.layer.constants.Keys;
import lombok.Data;

@Data
@Table("nativeAffinityOwner")
public class NativeAffinityOwner {
    @ColumnName("bindingId")
    @PrimaryKey(keyValue = Keys.UUID, order = "1")
    public String bindingId;
    @ColumnName("hostId")
    @PrimaryKey(keyValue = Keys.HOST_ID, order = "1")
    @PrimaryKey(keyValue = "nativeHostState", order = "1")
    public String hostId;
    @ColumnName("clusterId")
    public String clusterId;
    @ColumnName("state")
    @PrimaryKey(keyValue = "nativeHostState", order = "2")
    public String state;
    @ColumnName("files")
    public String files;
    @ColumnName("lastUsed")
    public Long lastUsed;
    @ColumnName("lastTask")
    public String lastTask;
    @ColumnName("lastPosition")
    public String lastPosition;
}

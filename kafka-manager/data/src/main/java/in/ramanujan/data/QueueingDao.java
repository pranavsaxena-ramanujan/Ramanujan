package in.ramanujan.data;

import in.ramanujan.base.pojo.CheckStatusQueueEvent;
import in.ramanujan.base.pojo.CheckStatusQueueEventWithMetadata;
import in.ramanujan.base.pojo.MachineAssignmentQueueEvent;
import in.ramanujan.base.pojo.MachineAssignmentTaskWithMetadata;
import io.vertx.core.Future;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public interface QueueingDao {
    public Future<Void> produce(final CheckStatusQueueEvent kafkaEvent);
    public Future<Void> produceMachineAssignment(final MachineAssignmentQueueEvent machineEvent);
    public Future<List<CheckStatusQueueEventWithMetadata>> consume();
    public Future<List<MachineAssignmentTaskWithMetadata>> consumeMachineAssignment();
    public Future<Void> subscribe();
    public Future<Void> subscribeMachineAssignment();
    public Future<Void> commit(Object metadata);
    public Future<Void> commitMachineAssignment(Object metadata);
}

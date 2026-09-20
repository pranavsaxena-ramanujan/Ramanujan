package in.ramanujan.shardedllm.model;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;

public class ProtocolModelTest {
    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    public void roundTripsFutureTrainingWorkWithoutSchemaChanges() throws Exception {
        TensorReference gradient = new TensorReference();
        gradient.setName("encoder.gradient");
        gradient.setDataType("bfloat16");
        gradient.setShape(Arrays.asList("micro_batch", "channels", "height", "width"));
        gradient.setByteLength(8192);
        gradient.setContentLocation("work/work-7/gradient.bin");
        gradient.setChecksum("sha256:example");

        WorkItem work = new WorkItem();
        work.setProtocolVersion("1.0");
        work.setWorkId("work-7");
        work.setSessionId("session-3");
        work.setCommand(WorkCommand.EXECUTE_ENTRYPOINT);
        work.setShardId("vision-encoder-2");
        work.setGraphNodeId("backward-encoder-2");
        work.setEntrypoint("backward");
        work.setCheckpointVersion(12);
        work.setInputs(Collections.singletonList(gradient));
        work.getAttributes().put("future.collective", "reduce-scatter");

        WorkItem restored = mapper.readValue(mapper.writeValueAsBytes(work), WorkItem.class);

        Assert.assertEquals("backward", restored.getEntrypoint());
        Assert.assertEquals("bfloat16", restored.getInputs().get(0).getDataType());
        Assert.assertEquals("reduce-scatter", restored.getAttributes().get("future.collective"));
    }

    @Test
    public void registrationAllowsMultipleDiskShardsWithOneResidentBudget() throws Exception {
        DeviceRegistration registration = new DeviceRegistration();
        registration.setProtocolVersion("1.0");
        registration.setDeviceId("worker-a");
        registration.setDiskCapacityBytes(4_000_000_000L);
        registration.setResidentBudgetBytes(1_000_000_000L);
        registration.setAssignedShardIds(Arrays.asList("phi3-00", "phi3-01"));
        registration.setCapabilities(Arrays.asList(StandardCapability.INFERENCE_DECODE,
                StandardCapability.CHECKPOINT_SAVE));

        DeviceRegistration restored = mapper.readValue(
                mapper.writeValueAsBytes(registration), DeviceRegistration.class);

        Assert.assertEquals(Arrays.asList("phi3-00", "phi3-01"), restored.getAssignedShardIds());
        Assert.assertTrue(restored.getResidentBudgetBytes() < restored.getDiskCapacityBytes());
    }
}
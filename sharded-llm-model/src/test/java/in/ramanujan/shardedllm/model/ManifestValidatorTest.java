package in.ramanujan.shardedllm.model;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;

public class ManifestValidatorTest {
    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    public void acceptsNonTransformerDagAndUnknownCapability() throws Exception {
        ModelPackageManifest manifest = baseManifest("spectrogram-autoencoder");
        manifest.setCapabilities(Arrays.asList("audio.encode", "future.training.custom"));

        ExecutionShardManifest encoder = shard("encoder", "programs/encoder.pb", "encode", "audio.encode");
        ExecutionShardManifest decoder = shard("decoder", "programs/decoder.pb", "decode", "audio.decode");
        manifest.setShards(Arrays.asList(encoder, decoder));

        ExecutionNode encode = node("encode", "encoder", "encode");
        ExecutionNode decode = node("decode", "decoder", "decode");
        decode.setDependsOn(Collections.singletonList("encode"));
        manifest.setExecutionGraph(Arrays.asList(encode, decode));

        String json = mapper.writeValueAsString(manifest);
        ModelPackageManifest restored = mapper.readValue(json, ModelPackageManifest.class);

        ManifestValidator.validateOrThrow(restored);
        Assert.assertTrue(restored.getCapabilities().contains("future.training.custom"));
        Assert.assertEquals("spectrogram-autoencoder", restored.getArchitectureId());
    }

    @Test
    public void acceptsSymbolicTensorShapesAndTrainingState() {
        ModelPackageManifest manifest = baseManifest("generic-decoder");
        ExecutionShardManifest shard = shard("block-a", "programs/block-a.pb", "forward", StandardCapability.FORWARD);

        TensorContract parameter = tensor("weight", "float16", "hidden", "intermediate");
        parameter.setRole("trainable-parameter");
        StateContract state = new StateContract();
        state.setTensor(parameter);
        state.setOwner("block-a");
        state.setPersistence("checkpoint");
        state.setMutable(true);
        state.setCheckpointed(true);
        shard.setStates(Collections.singletonList(state));
        shard.getEntrypoints().get(0).setStateNames(Collections.singletonList("weight"));

        manifest.setShards(Collections.singletonList(shard));
        manifest.setExecutionGraph(Collections.singletonList(node("forward", "block-a", "forward")));

        Assert.assertTrue(ManifestValidator.validate(manifest).isEmpty());
    }

    @Test
    public void rejectsCyclesAndAbsoluteArtifactPaths() {
        ModelPackageManifest manifest = baseManifest("invalid-model");
        ExecutionShardManifest shard = shard("only", "/tmp/program.pb", "run", "custom.run");
        manifest.setShards(Collections.singletonList(shard));
        ExecutionNode first = node("first", "only", "run");
        ExecutionNode second = node("second", "only", "run");
        first.setDependsOn(Collections.singletonList("second"));
        second.setDependsOn(Collections.singletonList("first"));
        manifest.setExecutionGraph(Arrays.asList(first, second));

        String errors = ManifestValidator.validate(manifest).toString();
        Assert.assertTrue(errors, errors.contains("package-relative path"));
        Assert.assertTrue(errors, errors.contains("cycle"));
    }

    private static ModelPackageManifest baseManifest(String architectureId) {
        ModelPackageManifest manifest = new ModelPackageManifest();
        manifest.setSchemaVersion("1.0");
        manifest.setPackageId(architectureId + "-test");
        manifest.setArchitectureId(architectureId);
        manifest.setArchitectureVersion("test");
        manifest.setSourceFormat("synthetic");
        return manifest;
    }

    private static ExecutionShardManifest shard(String shardId, String irPath,
                                                String entrypointName, String capability) {
        ExecutionEntrypoint entrypoint = new ExecutionEntrypoint();
        entrypoint.setName(entrypointName);
        entrypoint.setCommandId("command-" + entrypointName);
        entrypoint.setCapability(capability);
        entrypoint.setInputs(Collections.singletonList(tensor("input", "float16", "batch", "features")));
        entrypoint.setOutputs(Collections.singletonList(tensor("output", "float16", "batch", "features")));

        ExecutionShardManifest shard = new ExecutionShardManifest();
        shard.setShardId(shardId);
        shard.setIrPath(irPath);
        shard.setCapabilities(Collections.singletonList(capability));
        shard.setEntrypoints(Collections.singletonList(entrypoint));
        return shard;
    }

    private static ExecutionNode node(String nodeId, String shardId, String entrypoint) {
        ExecutionNode node = new ExecutionNode();
        node.setNodeId(nodeId);
        node.setShardId(shardId);
        node.setEntrypoint(entrypoint);
        return node;
    }

    private static TensorContract tensor(String name, String dataType, String... shape) {
        TensorContract tensor = new TensorContract();
        tensor.setName(name);
        tensor.setDataType(dataType);
        tensor.setShape(Arrays.asList(shape));
        return tensor;
    }
}
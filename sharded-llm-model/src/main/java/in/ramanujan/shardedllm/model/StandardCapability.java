package in.ramanujan.shardedllm.model;

public final class StandardCapability {
    public static final String INFERENCE_PREFILL = "inference.prefill";
    public static final String INFERENCE_DECODE = "inference.decode";
    public static final String FORWARD = "training.forward";
    public static final String BACKWARD = "training.backward";
    public static final String OPTIMIZER_STEP = "training.optimizer-step";
    public static final String CHECKPOINT_SAVE = "checkpoint.save";
    public static final String CHECKPOINT_RESTORE = "checkpoint.restore";

    private StandardCapability() {
    }
}
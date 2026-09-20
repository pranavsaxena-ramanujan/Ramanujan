package in.ramanujan.shardedllm.model;

public final class WorkCommand {
    public static final String ACTIVATE_SHARD = "activate-shard";
    public static final String EXECUTE_ENTRYPOINT = "execute-entrypoint";
    public static final String CHECKPOINT_STATE = "checkpoint-state";
    public static final String EVICT_SHARD = "evict-shard";
    public static final String CANCEL_SESSION = "cancel-session";

    private WorkCommand() {
    }
}
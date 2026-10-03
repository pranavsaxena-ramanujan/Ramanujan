package in.ramanujan.orchestrator.base;

public final class NativeSessionKey {
    private NativeSessionKey() {}

    private static String part(String value) {
        return value == null ? "-1:" : value.length() + ":" + value;
    }

    public static String binding(String cluster, String affinity, String session) {
        String binding = part(cluster) + part(affinity) + part(session);
        if (binding.length() > 512) throw new IllegalArgumentException("Native binding is too long");
        return binding;
    }

    public static String session(String cluster, String session) {
        return part(cluster) + part(session);
    }
}

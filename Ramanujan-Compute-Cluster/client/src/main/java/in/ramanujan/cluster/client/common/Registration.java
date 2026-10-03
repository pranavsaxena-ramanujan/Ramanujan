package in.ramanujan.cluster.client.common;

/** The bearer URL must never be printed or passed on a process command line. */
public final class Registration {
    public final String deviceId;
    public final String clusterId;
    public final String workerUrl;

    public Registration(String deviceId, String clusterId, String workerUrl) {
        this.deviceId = deviceId;
        this.clusterId = clusterId;
        this.workerUrl = workerUrl;
    }
}

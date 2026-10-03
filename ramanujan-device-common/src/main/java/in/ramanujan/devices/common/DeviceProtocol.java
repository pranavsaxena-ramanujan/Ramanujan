package in.ramanujan.devices.common;

import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;

/** Every device-to-orchestrator endpoint path and URL. Devices must not build these routes elsewhere. */
public final class DeviceProtocol {
    public static final String OPEN_PATH = "/pings/open";
    public static final String HEARTBEAT_PATH = "/pings/heartbeat";
    public static final String CAPACITY_PATH = "/pings/capacity";
    public static final String CHECKPOINT_PATH = "/pings/checkpoint";
    public static final String TASK_COMPLETE_PATH = "/task/complete";
    public static final String UPLOAD_BINARY_PATH = "/orchestrator/uploadBinary";
    public static final String BINARY_STAT_PATH = "/binary/stat";
    public static final String BINARY_FETCH_PATH = "/binary/fetch";
    public static final String DEBUG_VALUES_PATH = "/debugValues";

    private DeviceProtocol() {}

    public static String openUrl(String serverUrl, String deviceId, Integer affinityLimit) {
        if (affinityLimit != null && affinityLimit < 1) {
            throw new IllegalArgumentException("affinityLimit must be positive");
        }
        String url = trimSlash(serverUrl) + OPEN_PATH + "?uuid=" + encode(deviceId);
        return affinityLimit == null ? url : url + "&affinityLimit=" + affinityLimit;
    }

    public static String heartbeatUrl(String serverUrl, String deviceId, String taskId) {
        return trimSlash(serverUrl) + HEARTBEAT_PATH + "?uuid=" + encode(deviceId)
                + "&asyncId=" + encode(taskId);
    }

    public static String capacityUrl(String serverUrl) {
        return trimSlash(serverUrl) + CAPACITY_PATH;
    }

    public static String checkpointUrl(String serverUrl, String taskId) {
        return trimSlash(serverUrl) + CHECKPOINT_PATH + "?asyncId=" + encode(taskId);
    }

    public static String debugValuesUrl(String serverUrl, String taskId) {
        return trimSlash(serverUrl) + DEBUG_VALUES_PATH + "?asyncId=" + encode(taskId);
    }

    public static String taskCompleteUrl(String serverUrl) {
        return trimSlash(serverUrl) + TASK_COMPLETE_PATH;
    }

    public static String uploadBinaryUrl(String serverUrl, String taskId, String arrayId) {
        return trimSlash(serverUrl) + UPLOAD_BINARY_PATH + "?uuid=" + encode(taskId)
                + "&arrayId=" + encode(arrayId);
    }

    public static String binaryStatUrl(String serverUrl, String serverPath) {
        return trimSlash(serverUrl) + BINARY_STAT_PATH + "?path=" + encode(serverPath);
    }

    public static String binaryFetchUrl(String serverUrl, String serverPath) {
        return trimSlash(serverUrl) + BINARY_FETCH_PATH + "?path=" + encode(serverPath);
    }

    static String trimSlash(String value) {
        if (value == null || value.trim().isEmpty()) throw new IllegalArgumentException("serverUrl is required");
        return value.trim().replaceAll("/+$", "");
    }

    private static String encode(String value) {
        if (value == null || value.isEmpty()) throw new IllegalArgumentException("ping identity is required");
        try {
            return URLEncoder.encode(value, "UTF-8");
        } catch (UnsupportedEncodingException e) {
            throw new IllegalStateException("UTF-8 is unavailable", e);
        }
    }
}

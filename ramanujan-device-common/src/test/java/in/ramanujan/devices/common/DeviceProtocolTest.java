package in.ramanujan.devices.common;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class DeviceProtocolTest {
    @Test
    public void buildsScopedGatewayPingUrlsAndEncodesIdentities() {
        assertEquals("https://portal.example/worker/token/pings/open?uuid=device%2Fone&affinityLimit=3",
                DeviceProtocol.openUrl("https://portal.example/worker/token/", "device/one", 3));
        assertEquals("https://portal.example/worker/token/pings/heartbeat?uuid=device+one&asyncId=task%2Fone",
                DeviceProtocol.heartbeatUrl("https://portal.example/worker/token", "device one", "task/one"));
        assertEquals("https://portal.example/worker/token/pings/capacity",
                DeviceProtocol.capacityUrl("https://portal.example/worker/token/"));
    }

    @Test
    public void buildsTaskAndBinaryUrls() {
        String gateway = "https://portal.example/worker/token";
        assertEquals(gateway + "/task/complete", DeviceProtocol.taskCompleteUrl(gateway + "/"));
        assertEquals(gateway + "/orchestrator/uploadBinary?uuid=t+1&arrayId=a%261",
                DeviceProtocol.uploadBinaryUrl(gateway, "t 1", "a&1"));
        assertEquals(gateway + "/binary/stat?path=%2Fmodels%2Fq.bin",
                DeviceProtocol.binaryStatUrl(gateway, "/models/q.bin"));
        assertEquals(gateway + "/binary/fetch?path=%2Fmodels%2Fq.bin",
                DeviceProtocol.binaryFetchUrl(gateway, "/models/q.bin"));
        assertEquals(gateway + "/pings/checkpoint?asyncId=t1", DeviceProtocol.checkpointUrl(gateway, "t1"));
        assertEquals(gateway + "/debugValues?asyncId=t1_0", DeviceProtocol.debugValuesUrl(gateway, "t1_0"));
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsMissingServer() {
        DeviceProtocol.taskCompleteUrl(" ");
    }
}

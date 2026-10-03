package in.ramanujan.cluster.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import in.ramanujan.cluster.client.common.*;
import org.junit.Test;
import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;
import static org.junit.Assert.*;

public class ClientContractTest {
    @Test public void joinUsesRoomContractAndAcceptsOnlySamePortalScopedUrls() throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        String portal = "http://127.0.0.1:" + server.getAddress().getPort();
        AtomicReference<Map> submitted = new AtomicReference<>();
        server.createContext("/api/devices/join", exchange -> {
            assertEquals("POST", exchange.getRequestMethod());
            submitted.set(new ObjectMapper().readValue(exchange.getRequestBody(), Map.class));
            byte[] body = ("{\"deviceId\":\"d1\",\"clusterId\":\"c1\",\"workerUrl\":\"" + portal + "/worker/test_token\"}").getBytes("UTF-8");
            exchange.sendResponseHeaders(201, body.length);
            try (OutputStream out = exchange.getResponseBody()) { out.write(body); }
        });
        server.start();
        try {
            Registration joined = new JoinClient().join(portal + "/", "room", "private secret", "desktop", "linux", true);
            assertEquals(portal + "/worker/test_token", joined.workerUrl);
            assertEquals("d1", joined.deviceId);
            assertEquals(new HashSet<>(Arrays.asList("roomId", "joinSecret", "name", "platform")), submitted.get().keySet());
            assertEquals("private secret", submitted.get().get("joinSecret"));
            for (String invalid : Arrays.asList("http://elsewhere/worker/token", portal + "/worker/token/other",
                    portal + "/task/complete", portal + "/worker/token?key=secret", portal + "/worker/token#fragment")) {
                try { JoinClient.validateWorkerUrl(portal, invalid); fail("must reject unscoped URL"); }
                catch (IOException expected) { }
            }

            assertFalse(JoinClient.redact("GET " + joined.workerUrl + "/pings/open").contains("test_token"));
        } finally { server.stop(0); }
    }

    @Test public void productionRequiresHttpsAndDevelopmentHttpIsExplicitAndLocal() {
        JoinClient.validateTransport("https://portal.example", false);
        for (String local : Arrays.asList("localhost", "127.0.0.1", "10.1.2.3", "172.16.2.3",
                "192.168.2.3", "[::1]", "[fd00::1]")) {
            JoinClient.validateTransport("http://" + local, true);
            try { JoinClient.validateTransport("http://" + local, false); fail("HTTP must be explicitly enabled"); }
            catch (IllegalArgumentException expected) { }
        }
        for (String publicHost : Arrays.asList("portal.example", "8.8.8.8", "172.32.1.1", "10.999.1.1", "[2001:4860::1]")) {
            try { JoinClient.validateTransport("http://" + publicHost, true); fail("Public HTTP forbidden"); }
            catch (IllegalArgumentException expected) { }
        }
    }

    @Test public void configPersistsReconnectButNotJoinSecret() throws Exception {
        Path root = fixture();
        try {
            ConfigStore store = new ConfigStore(root.resolve("config"));
            store.save("https://portal.example", "room1", "desktop",
                    new Registration("d1", "c1", "https://portal.example/worker/test_token"));
            Properties values = store.load();
            assertEquals("room1", values.getProperty("roomId"));
            assertEquals("https://portal.example/worker/test_token", values.getProperty("workerUrl"));
            assertFalse(values.containsKey("joinSecret"));
            if (Files.getFileAttributeView(root.resolve("config/device.properties"), java.nio.file.attribute.PosixFileAttributeView.class) != null) {
                assertEquals("rw-------", java.nio.file.attribute.PosixFilePermissions.toString(
                        Files.getPosixFilePermissions(root.resolve("config/device.properties"))));
            }
        } finally { cleanup(root); }
    }

    @Test public void launchKeepsBearerOutOfArgumentsAndRequiresBothLibraries() throws Exception {
        Path root = fixture();
        try {
            Path bundle = root.resolve("bundle"); Files.createDirectories(bundle.resolve("native"));
            Files.createFile(bundle.resolve("developer-console.jar"));
            Files.createFile(bundle.resolve("native").resolve(System.mapLibraryName("native")));
            String url = "https://portal.example/worker/test_token";
            try { WorkerLauncher.command(bundle, root.resolve("cache"), url); fail("LLM library required"); }
            catch (IOException expected) { }
            Files.createFile(bundle.resolve("native").resolve(System.mapLibraryName("ramanujan_llm")));
            ProcessBuilder command = WorkerLauncher.command(bundle, root.resolve("cache"), url);
            assertFalse(command.command().toString().contains("test_token"));
            assertEquals(url, command.environment().get("RAMANUJAN_WORKER_URL"));
            assertEquals(Arrays.asList("worker", "1", "--cache", root.resolve("cache").toString(),
                    "--llm-sessions", "8"), command.command().subList(3, command.command().size()));
        } finally { cleanup(root); }
    }

    private Path fixture() throws IOException {
        Path root = Paths.get("target", "test-fixtures", UUID.randomUUID().toString());
        Files.createDirectories(root); return root;
    }
    private void cleanup(Path root) throws IOException {
        try (java.util.stream.Stream<Path> files = Files.walk(root)) {
            files.sorted(Comparator.reverseOrder()).forEach(file -> file.toFile().delete());
        }
    }
}

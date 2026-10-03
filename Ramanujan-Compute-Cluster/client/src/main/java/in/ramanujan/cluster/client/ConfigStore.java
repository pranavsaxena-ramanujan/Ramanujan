package in.ramanujan.cluster.client;

import in.ramanujan.cluster.client.common.JoinClient;
import in.ramanujan.cluster.client.common.Registration;
import java.io.*;
import java.nio.file.*;
import java.nio.file.attribute.*;
import java.util.*;

public final class ConfigStore {
    private final Path directory;
    public ConfigStore(Path directory) { this.directory = directory; }

    public Properties load() throws IOException {
        Properties values = new Properties();
        Path file = directory.resolve("device.properties");
        if (Files.exists(file)) {
            secure(directory); secure(file);
            try (InputStream in = Files.newInputStream(file)) { values.load(in); }
            try { JoinClient.validateTransport(values.getProperty("portal", ""), Boolean.parseBoolean(values.getProperty("allowLocalHttp", "false"))); }
            catch (IllegalArgumentException e) { throw new IOException("Saved portal requires HTTPS or explicit local development opt-in."); }
            JoinClient.validateWorkerUrl(values.getProperty("portal", ""), values.getProperty("workerUrl", ""));
        }
        return values;
    }

    public void save(String portal, String room, String name, Registration joined) throws IOException {
        save(portal, room, name, joined, false);
    }

    public void save(String portal, String room, String name, Registration joined, boolean allowLocalHttp) throws IOException {
        JoinClient.validateTransport(portal, allowLocalHttp);
        JoinClient.validateWorkerUrl(portal, joined.workerUrl);
        Files.createDirectories(directory); secure(directory);
        Properties values = new Properties();
        values.setProperty("portal", portal); values.setProperty("roomId", room);
        values.setProperty("name", name); values.setProperty("deviceId", joined.deviceId);
        values.setProperty("clusterId", joined.clusterId); values.setProperty("workerUrl", joined.workerUrl);
        values.setProperty("allowLocalHttp", Boolean.toString(allowLocalHttp));
        Path partial = directory.resolve("device.properties.partial");
        Files.deleteIfExists(partial); Files.createFile(partial); secure(partial);
        try {
            try (OutputStream out = Files.newOutputStream(partial)) { values.store(out, "Private device registration; do not share"); }
            Files.move(partial, directory.resolve("device.properties"), StandardCopyOption.REPLACE_EXISTING);
            secure(directory.resolve("device.properties"));
        } finally { Files.deleteIfExists(partial); }
    }

    private static void secure(Path path) throws IOException {
        PosixFileAttributeView posix = Files.getFileAttributeView(path, PosixFileAttributeView.class);
        if (posix != null) {
            posix.setPermissions(PosixFilePermissions.fromString(Files.isDirectory(path) ? "rwx------" : "rw-------"));
            return;
        }
        AclFileAttributeView acl = Files.getFileAttributeView(path, AclFileAttributeView.class);
        if (acl == null) throw new IOException("Private credential storage requires POSIX permissions or Windows ACLs.");
        AclEntry entry = AclEntry.newBuilder().setType(AclEntryType.ALLOW).setPrincipal(acl.getOwner())
                .setPermissions(EnumSet.allOf(AclEntryPermission.class)).build();
        acl.setAcl(Collections.singletonList(entry));
    }
}

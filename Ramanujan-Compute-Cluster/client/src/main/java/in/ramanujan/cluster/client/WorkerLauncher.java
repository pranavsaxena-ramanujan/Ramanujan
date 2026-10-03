package in.ramanujan.cluster.client;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;

public final class WorkerLauncher {
    public static ProcessBuilder command(Path bundle, Path cache, String workerUrl) throws IOException {
        String windows = System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("win") ? ".exe" : "";
        Path javaExecutable = Paths.get(System.getProperty("java.home"), "bin", "java" + windows);
        Path worker = bundle.resolve("developer-console.jar");
        Path nativeDir = bundle.resolve("native");
        for (Path path : Arrays.asList(javaExecutable, worker, nativeDir.resolve(System.mapLibraryName("native")),
                nativeDir.resolve(System.mapLibraryName("ramanujan_llm")))) {
            if (!Files.isRegularFile(path)) throw new IOException("Missing bundled runtime or worker/native artifact: " + path.getFileName());
        }
        Files.createDirectories(cache);
        ProcessBuilder builder = new ProcessBuilder(javaExecutable.toString(), "-jar", worker.toString(), "worker", "1",
                "--cache", cache.toString(), "--llm-sessions", "8");
        builder.environment().put("RAMANUJAN_WORKER_URL", workerUrl);
        builder.environment().put("RAMANUJAN_WS", nativeDir.toAbsolutePath().toString());
        builder.environment().put("TMPDIR", cache.toAbsolutePath().toString());
        String pathKey = "PATH";
        for (String key : builder.environment().keySet()) if (key.equalsIgnoreCase("PATH")) pathKey = key;
        builder.environment().put(pathKey, nativeDir.toAbsolutePath() + java.io.File.pathSeparator
                + builder.environment().getOrDefault(pathKey, ""));
        for (String key : Arrays.asList("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")) {
            builder.environment().put(key, nativeDir.toAbsolutePath() + java.io.File.pathSeparator
                    + builder.environment().getOrDefault(key, ""));
        }
        builder.redirectErrorStream(true);
        return builder;
    }
}

package in.ramanujan.developer.console.operationImpl;

import in.ramanujan.developer.console.model.pojo.csv.CsvInformation;
import org.junit.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

public class ExecuteInlineServerShardTest {
    private static CsvInformation csv(String path, String data) {
        CsvInformation info = new CsvInformation();
        info.setFileName(path);
        info.setData(data);
        return info;
    }

    @Test
    public void compiledProgramKeyTracksInputShapesNotLocations() {
        String prompt11 = ExecuteInlineServer.inputShapeKey(Arrays.asList(
                csv("/run-a/hidden.csv", "0,0,0\n0\n"), csv("/run-a/params.csv", "11.0\n")));
        String prompt11Elsewhere = ExecuteInlineServer.inputShapeKey(Arrays.asList(
                csv("/run-b/params.csv", "12.0\n"), csv("/run-b/hidden.csv", "0,0,0\n0\n")));
        String prompt12 = ExecuteInlineServer.inputShapeKey(Arrays.asList(
                csv("/run-a/hidden.csv", "0,0,0\n0\n0\n"), csv("/run-a/params.csv", "11.0\n")));

        assertEquals(prompt11, prompt11Elsewhere);
        assertNotEquals(prompt11, prompt12);
    }
    @Test
    public void registersProgramsFromDiskWithoutLoadingWeights() throws IOException {
        Path root = Files.createTempDirectory("shard-package-");
        Path shard = root.resolve("shard-00");
        Path programs = shard.resolve("programs");
        Files.createDirectories(programs);
        Files.write(shard.resolve("manifest.json"), new byte[0]);
        Files.write(programs.resolve("prefill.py"), new byte[0]);
        Files.write(programs.resolve("decode.py"), new byte[0]);
        Path manifest = root.resolve("model-manifest.json");
        Files.write(manifest,
                "{\"shards\":[{\"manifestPath\":\"shard-00/manifest.json\"}]}"
                        .getBytes(StandardCharsets.UTF_8));

        Set<Path> paths = new ExecuteInlineServer().readShardPrograms(manifest);
        assertEquals(2, paths.size());
        assertTrue(paths.contains(programs.resolve("decode.py").toRealPath()));
    }

    @Test(expected = IOException.class)
    public void rejectsShardOutsidePackage() throws IOException {
        Path root = Files.createTempDirectory("shard-package-");
        Path external = Files.createTempFile("external-shard-", ".json");
        Path manifest = root.resolve("model-manifest.json");
        String relativePath = root.relativize(external).toString();
        Files.write(manifest,
                ("{\"shards\":[{\"manifestPath\":\"" + relativePath + "\"}]}")
                        .getBytes(StandardCharsets.UTF_8));
        new ExecuteInlineServer().readShardPrograms(manifest);
    }
}
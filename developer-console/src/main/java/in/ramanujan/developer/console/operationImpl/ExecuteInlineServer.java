package in.ramanujan.developer.console.operationImpl;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import in.ramanujan.developer.console.Operation;
import in.ramanujan.developer.console.model.pojo.CodeRunRequest;
import in.ramanujan.developer.console.model.pojo.csv.CsvInformation;
import in.ramanujan.pojo.RuleEngineInput;
import in.ramanujan.pojo.ruleEngineInputUnitsExt.Variable;
import in.ramanujan.pojo.ruleEngineInputUnitsExt.array.Array;
import in.ramanujan.translation.codeConverter.CodeSnippetElement;
import in.ramanujan.translation.codeConverter.DagElement;
import in.ramanujan.translation.codeConverter.exception.CompilationException;
import in.ramanujan.translation.codeConverter.grammar.debugLevelCodeCreatorImpl.ActualDebugCodeCreator;
import in.ramanujan.translation.codeConverter.pojo.ExtractedCodeAndFunctionCode;
import in.ramanujan.translation.codeConverter.pojo.TranslateResponse;
import in.ramanujan.translation.codeConverter.utils.TranslateUtil;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import static in.ramanujan.developer.console.operationImpl.ExecutorImpl.createJson;

/**
 * Persistent server mode: one JVM stays alive for the entire inference run.
 * Reads newline-delimited commands from stdin:
 *
 *   run <kernel.py> <csv1> <csv2> ...   — compile and execute a kernel
 *   REGISTER_SHARDS <model-manifest.json> — register package program paths
 *   CHANGE_SHARD <next-program.py>       — evict the previous shard after dumps
 *   EVICT_WEIGHTS                        — unmap cached immutable weight binaries
 *   dump <name> [file]                  — dump array to stdout or file
 *   var  <name>                         — print scalar variable value
 *   arr  <name> <index>                 — print one array element
 *   quit                                — exit cleanly
 *
 * After each `run` command completes, prints exactly one line: "KERNEL_DONE"
 * After each `dump <name> <file>`, prints: "Dumped <name> (<size>) to <file>"
 * On error, prints: "KERNEL_ERROR: <message>"
 *
 * Benefits vs per-call JVM:
 *   - JVM startup paid once (~5-8s)
 *   - OpenCL context initialised once (clBuildProgram cached in s_programCache)
 *   - No process spawn overhead between layers
 */
public class ExecuteInlineServer extends ExecuteInline {

    private static final class CompiledProgram {
        final DagElement firstDag;
        final List<DagElement> dagList;
        final Map<String, Variable> variableMap;
        final Map<String, Array> arrayMap;

        CompiledProgram(DagElement firstDag, List<DagElement> dagList,
                        Map<String, Variable> variableMap, Map<String, Array> arrayMap) {
            this.firstDag = firstDag;
            this.dagList = dagList;
            this.variableMap = variableMap;
            this.arrayMap = arrayMap;
        }
    }

    // Unregistered runs keep a few recent programs so alternating kernels (embed, layer types, head)
    // are not recompiled on every switch.
    static final int MAX_UNREGISTERED_PROGRAMS = 8;

    // Keyed by program path and input shapes; holds every registered shard program, else the
    // most recently used MAX_UNREGISTERED_PROGRAMS entries.
    private final Map<String, CompiledProgram> compiledPrograms = new LinkedHashMap<String, CompiledProgram>(16, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<String, CompiledProgram> eldest) {
            return registeredShardPrograms == null && size() > MAX_UNREGISTERED_PROGRAMS;
        }
    };
    private String nextShardKernel = null;
    private boolean lastRunSucceeded = true;
    private Set<Path> registeredShardPrograms = null;

    // Default set of array names always populated into the dump store.
    // Existing orchestrators that send no --dump flags rely on this list.
    // New orchestrators can extend it per-call via: run <kernel> <csvs> --dump name1 name2 ...
    // Note: endsWith patterns ("_k_cache", "_v_cache") are handled separately below
    // because a plain Set cannot store suffix patterns.
    private static final Set<String> DEFAULT_DUMP_TARGETS = new HashSet<>(Arrays.asList(
            "generated_tokens", "hidden", "debug_out",
            "k_cache", "v_cache", "argmax_arr",
            "h_state", "step_arr", "cur_n_seq_arr"
    ));

    @Override
    public void execute(List<String> args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

        // Signal readiness so the Python orchestrator knows the JVM is up
        System.out.println("SERVER_READY");
        System.out.flush();

        String line;
        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;

            if (line.equalsIgnoreCase("quit") || line.equalsIgnoreCase("exit")) {
                System.out.println("SERVER_EXIT");
                System.out.flush();
                break;
            }

            if (line.startsWith("REGISTER_SHARDS ")) {
                try {
                    registeredShardPrograms = readShardPrograms(
                            Paths.get(line.substring("REGISTER_SHARDS ".length()).trim()));
                        new in.ramanujan.rule.engine.NativeProcessor().resetShardSession();
                    compiledPrograms.clear();
                    System.out.println("SHARDS_REGISTERED");
                } catch (Exception e) {
                    System.out.println("SHARD_ERROR: " + e.getMessage());
                }
                System.out.flush();
                continue;
            }

            if (line.equals("EVICT_WEIGHTS")) {
                try {
                    new in.ramanujan.rule.engine.NativeProcessor().changeShard();
                    System.out.println("WEIGHTS_EVICTED");
                } catch (Exception e) {
                    System.out.println("SHARD_ERROR: " + e.getMessage());
                }
                System.out.flush();
                continue;
            }

            if (line.startsWith("CHANGE_SHARD ")) {
                String nextKernel = line.substring("CHANGE_SHARD ".length()).trim();
                try {
                    Path nextProgram = Paths.get(nextKernel).toRealPath();
                    if (!lastRunSucceeded || registeredShardPrograms == null
                            || !registeredShardPrograms.contains(nextProgram)) {
                        throw new IllegalStateException("restart worker or register the next program");
                    }
                    new in.ramanujan.rule.engine.NativeProcessor().changeShard();
                    ExecutorImpl.setStores(null, null);
                    ExecutorImpl.setBinaryArrayFileStore(null);
                    nextShardKernel = nextProgram.toString();
                    System.out.println("SHARD_READY");
                } catch (Exception e) {
                    System.out.println("SHARD_ERROR: " + e.getMessage());
                }
                System.out.flush();
                continue;
            }

            if (line.startsWith("run ")) {
                // Parse: run <kernel.py> <csv1> ... [--dump name1 name2 ...]
                // Everything after --dump is a set of array names the client wants populated.
                String[] parts = line.split("\\s+");
                List<String> kernelArgs = new ArrayList<>();
                // Seed with defaults so existing orchestrators (no --dump) keep working.
                Set<String> dumpTargets = new HashSet<>(DEFAULT_DUMP_TARGETS);
                boolean inDump = false;
                for (int i = 1; i < parts.length; i++) {
                    if (parts[i].equals("--dump")) { inDump = true; continue; }
                    if (inDump) dumpTargets.add(parts[i]);
                    else        kernelArgs.add(parts[i]);
                }
                try {
                    if (registeredShardPrograms != null && !registeredShardPrograms.contains(
                            Paths.get(kernelArgs.get(0)).toRealPath())) {
                        throw new IllegalArgumentException("program is not in the registered package");
                    }
                    if (nextShardKernel != null && !nextShardKernel.equals(
                            Paths.get(kernelArgs.get(0)).toRealPath().toString())) {
                        throw new IllegalArgumentException("expected shard program " + nextShardKernel);
                    }
                    lastRunSucceeded = false;
                    runKernel(kernelArgs, dumpTargets);
                    lastRunSucceeded = true;
                    nextShardKernel = null;
                    System.out.println("KERNEL_DONE");
                } catch (Exception e) {
                    System.out.println("KERNEL_ERROR: " + e.getMessage());
                }
                System.out.flush();
                continue;
            }

            // Query commands (dump / var / arr) reuse shared helper
            handleQueryCommand(line);
            System.out.flush();
        }
    }

    Set<Path> readShardPrograms(Path manifestPath) throws IOException {
        Path root = manifestPath.toRealPath().getParent();
        JsonNode shards = new ObjectMapper().readTree(manifestPath.toFile()).path("shards");
        if (!shards.isArray() || shards.size() == 0) {
            throw new IOException("package has no shards");
        }
        Set<Path> programs = new HashSet<>();
        for (JsonNode shard : shards) {
            String relativePath = shard.path("manifestPath").asText();
            if (relativePath.isEmpty()) {
                throw new IOException("shard has no manifestPath");
            }
            Path shardManifest = root.resolve(relativePath).toRealPath();
            if (!shardManifest.startsWith(root) || !Files.isRegularFile(shardManifest)) {
                throw new IOException("invalid shard manifest: " + relativePath);
            }
            for (String entrypoint : Arrays.asList("prefill", "decode")) {
                Path program = shardManifest.getParent().resolve("programs")
                        .resolve(entrypoint + ".py").toRealPath();
                if (!program.startsWith(root) || !Files.isRegularFile(program)) {
                    throw new IOException("invalid shard program: " + program);
                }
                programs.add(program);
            }
            Path residentProgram = shardManifest.getParent().resolve("programs/decode_resident.py");
            if (Files.exists(residentProgram)) {
                Path resolved = residentProgram.toRealPath();
                if (!resolved.startsWith(root) || !Files.isRegularFile(resolved)) {
                    throw new IOException("invalid resident shard program: " + residentProgram);
                }
                programs.add(resolved);
            }
            Path fusedProgram = shardManifest.getParent().resolve("programs/decode_fused.py");
            if (Files.exists(fusedProgram)) {
                Path resolved = fusedProgram.toRealPath();
                if (!resolved.startsWith(root) || !Files.isRegularFile(resolved)) {
                    throw new IOException("invalid fused shard program: " + fusedProgram);
                }
                programs.add(resolved);
            }
        }
        return programs;
    }

    static String inputShapeKey(List<CsvInformation> csvList) {
        List<String> shapes = new ArrayList<>();
        for (CsvInformation csv : csvList) {
            String data = csv.getData() == null ? "" : csv.getData();
            int lineEnd = data.indexOf('\n');
            String firstLine = lineEnd < 0 ? data : data.substring(0, lineEnd);
            int columns = firstLine.isEmpty() ? 0 : firstLine.split(",", -1).length;
            int rows = 0;
            for (int i = 0; i < data.length(); i++) {
                if (data.charAt(i) == '\n') rows++;
            }
            if (!data.isEmpty() && !data.endsWith("\n")) rows++;
            shapes.add(Paths.get(csv.getFileName()).getFileName() + ":" + rows + "x" + columns);
        }
        Collections.sort(shapes);
        return String.join(",", shapes);
    }

    // -------------------------------------------------------------------------
    // Core kernel execution (mirrors ExecuteInline.execute but no console loop)
    // -------------------------------------------------------------------------

    private void runKernel(List<String> args, Set<String> dumpTargets) throws IOException, CompilationException {
        long t0 = System.currentTimeMillis();

        System.err.println("[Server] run: " + args.get(0) + " +" + (args.size() - 1) + " CSVs");

        CodeRunRequest req = createJson(args);
        System.err.println("[Server] createJson: " + (System.currentTimeMillis() - t0) + "ms");
        String kernelPath = args.get(0);
        List<CsvInformation> csvList = req.getCsvInformationList() != null
                ? req.getCsvInformationList() : new ArrayList<>();

        Map<String, Variable> variableMap;
        Map<String, Array>    arrayMap;
        DagElement            firstDag;
        List<DagElement>      dagList;

        String programKey = kernelPath + "|" + inputShapeKey(csvList);
        CompiledProgram compiled = compiledPrograms.get(programKey);
        if (compiled != null) {
            variableMap = compiled.variableMap;
            arrayMap    = compiled.arrayMap;
            firstDag    = compiled.firstDag;
            dagList     = compiled.dagList;

            // Clear mutable state so re-population starts clean
            for (Array a : arrayMap.values()) {
                if (a.getValues() != null) a.getValues().clear();
                a.setBinaryFile(null);
            }
            for (Variable v : variableMap.values()) {
                v.setValue(null);
            }

            long repopStart = System.currentTimeMillis();
            translateUtil.repopulateCsvArrayValues(arrayMap, csvList);
            System.err.println("[Server] compiled in (cached) " + (System.currentTimeMillis() - t0)
                    + "ms  repopulate=" + (System.currentTimeMillis() - repopStart) + "ms");
        } else {
            variableMap = new HashMap<>();
            arrayMap    = new HashMap<>();

            String code = req.getCode();
            Map<String, RuleEngineInput> functionCallsRuleEngineInput = new HashMap<>();
            ActualDebugCodeCreator debugCreator = new ActualDebugCodeCreator("", 0);

            String extractedCode;
            int linesForFunctions;

            if (TranslateUtil.isPythonCode(code)) {
                extractedCode = code;
                linesForFunctions = 0;
            } else {
                ExtractedCodeAndFunctionCode extracted =
                        translateUtil.extractCodeWithoutAbstractCodeDeclaration(
                                code, functionCallsRuleEngineInput, debugCreator);
                for (Map.Entry<String, RuleEngineInput> e : functionCallsRuleEngineInput.entrySet()) {
                    for (Variable v : e.getValue().getVariables()) variableMap.put(v.getId(), v);
                    for (Array a : e.getValue().getArrays())        arrayMap.put(a.getId(), a);
                }
                extractedCode = extracted.getExtractedCode();
                linesForFunctions = debugCreator.getLine();
            }

            CodeSnippetElement firstSnippet = translateUtil.getCodeSnippets(
                    extractedCode, new HashMap<>(), new HashMap<>(), new HashMap<>());

            dagList = new ArrayList<>();
            Map<String, String> dagCodeMap = new HashMap<>();
            firstDag = translateUtil.populateAllDagElements(
                    firstSnippet, csvList, functionCallsRuleEngineInput,
                    variableMap, arrayMap, dagList, dagCodeMap, linesForFunctions);

            System.err.println("[Server] compiled in " + (System.currentTimeMillis() - t0)
                    + "ms  DAG=" + (dagList.size() + 1));

            compiledPrograms.put(programKey,
                    new CompiledProgram(firstDag, dagList, variableMap, arrayMap));
        }

        long execStart = System.currentTimeMillis();
        boolean sequential = "true".equalsIgnoreCase(System.getenv("RAMANUJAN_SEQUENTIAL"));
        if (sequential) {
            executeSequentially(firstDag, dagList, variableMap, arrayMap);
        } else {
            executeInParallel(firstDag, dagList, variableMap, arrayMap);
        }
        System.err.println("[Server] executed in " + (System.currentTimeMillis() - execStart) + "ms");

        // Build stores so query commands can access results
        Map<String, Object> varStore = new HashMap<>();
        for (Variable v : variableMap.values()) varStore.put(v.getName(), v.getValue());


        Map<String, Map<String, Object>> arrStore = new HashMap<>();
        // Populate only arrays in dumpTargets (always non-empty: seeded from DEFAULT_DUMP_TARGETS
        // plus any extra names the client added via --dump).
        // endsWith patterns for per-layer KV caches are checked separately.
        for (Array a : arrayMap.values()) {
            String id = a.getId();
            if (id.contains("func") || !id.contains("_name_")) continue;
            String name = id.split("_name_")[1];
            if (!dumpTargets.contains(name)
                    && !name.endsWith("_k_cache")
                    && !name.endsWith("_v_cache")) continue;

            Map<String, Object> vals = a.getValues();
            if (vals == null) continue;
            for (Map.Entry<String, Object> e : vals.entrySet()) {
                Map<String, Object> m = arrStore.computeIfAbsent(name, k -> new HashMap<>());
                m.put(e.getKey(), e.getValue());
            }
        }

        ExecutorImpl.setStores(varStore, arrStore);
        // Large stub strings from earlier kernels can pile up into GC storms, so collect
        // eagerly once the heap is under pressure; a full GC on every run costs several ms.
        if (!"true".equalsIgnoreCase(System.getenv("RAMANUJAN_RESIDENT_KV")) && heapUnderPressure()) {
            System.gc();
        }
    }

    static final double GC_HEAP_FRACTION = 0.6;

    private static boolean heapUnderPressure() {
        Runtime runtime = Runtime.getRuntime();
        long used = runtime.totalMemory() - runtime.freeMemory();
        return used > runtime.maxMemory() * GC_HEAP_FRACTION;
    }

    // -------------------------------------------------------------------------
    // Query command handler (dump / var / arr)
    // -------------------------------------------------------------------------

    private void handleQueryCommand(String line) {
        if (line.startsWith("var ")) {
            String[] p = line.split(" ", 2);
            if (p.length == 2) {
                Object v = ExecutorImpl.variableStore.get(p[1]);
                System.out.println(v != null ? p[1] + " = " + v : "Variable not found.");
            }
            return;
        }

        if (line.startsWith("arr ")) {
            String[] p = line.split(" ");
            if (p.length == 3) {
                Map<String, Object> arr = ExecutorImpl.arrayStore.get(p[1]);
                if (arr != null && arr.containsKey(p[2])) {
                    System.out.println(p[1] + "[" + p[2] + "] = " + arr.get(p[2]));
                } else {
                    System.out.println("Array or index not found.");
                }
            }
            return;
        }

        if (line.startsWith("dump_diff ")) {
            // dump_diff <arrayName> <startIdx> <endIdxInclusive> [outputFile]
            // Emits only entries whose flat index falls in [startIdx, endIdxInclusive],
            // one "key,value" per line. The orchestrator chooses the range based on
            // which rows the shard touched, so no JVM snapshot is needed.
            String[] p = line.split(" ");
            if (p.length < 4) {
                System.out.println("Usage: dump_diff <arrayName> <startIdx> <endIdxInclusive> [outputFile]");
                return;
            }
            String arrName = p[1];
            long startIdx, endIdx;
            try {
                startIdx = Long.parseLong(p[2]);
                endIdx   = Long.parseLong(p[3]);
            } catch (NumberFormatException nfe) {
                System.out.println("dump_diff: startIdx/endIdx must be integers");
                return;
            }
            String outFile = p.length >= 5 ? p[4] : null;
            Map<String, Object> arr = ExecutorImpl.arrayStore.get(arrName);
            if (arr == null) {
                System.out.println("Array not found: " + arrName);
                return;
            }
            StringBuilder sb = new StringBuilder();
            int count = 0;
            for (Map.Entry<String, Object> e : arr.entrySet()) {
                String key = e.getKey();
                long flat;
                try {
                    int us = key.indexOf('_');
                    if (us < 0) {
                        flat = Long.parseLong(key);
                    } else {
                        // 2D key "row_col" → flatten via row * (maxCol+1) + col is unknown here;
                        // instead, treat the key itself as opaque and let the orchestrator
                        // reverse-map it. For the range filter we use a fallback: only filter
                        // 1D-style keys; for 2D keys, emit unconditionally (rare in our use).
                        sb.append(key).append(',').append(e.getValue()).append('\n');
                        count++;
                        continue;
                    }
                } catch (NumberFormatException nfe) {
                    continue;
                }
                if (flat >= startIdx && flat <= endIdx) {
                    sb.append(key).append(',').append(e.getValue()).append('\n');
                    count++;
                }
            }
            if (outFile != null) {
                try {
                    Files.write(Paths.get(outFile),
                            sb.toString().getBytes(StandardCharsets.UTF_8));
                    System.out.println("Dumped diff " + arrName + " (" + count + ") to " + outFile);
                } catch (Exception ex) {
                    System.out.println("Error writing file: " + ex.getMessage());
                }
            } else {
                System.out.print(sb.toString());
                System.out.println("Dumped diff " + arrName + " (" + count + ")");
            }
            return;
        }

        if (line.startsWith("take ")) {
            String[] parts = line.split(" ", 3);
            if (parts.length != 3 || !ExecutorImpl.binaryArrayFileStore.containsKey(parts[1])) {
                System.out.println("SHARD_ERROR: binary array not found");
                return;
            }
            try {
                takeBinaryArray(ExecutorImpl.binaryArrayFileStore.get(parts[1]), parts[2]);
                ExecutorImpl.binaryArrayFileStore.remove(parts[1]);
                System.out.println("Taken " + parts[1]);
            } catch (IOException ex) {
                System.out.println("SHARD_ERROR: " + ex.getMessage());
            }
            return;
        }

        if (line.startsWith("dump ")) {
            String[] p = line.split(" ");
            String arrName = p.length >= 2 ? p[1] : null;
            String outFile = p.length >= 3 ? p[2] : null;
            if (arrName == null) {
                System.out.println("Usage: dump <arrayName> [outputFile]");
                return;
            }
            String binaryFile = ExecutorImpl.binaryArrayFileStore.get(arrName);
            if (binaryFile != null && outFile != null) {
                try {
                    dumpBinaryArray(binaryFile, outFile);
                    System.out.println("Dumped " + arrName + " to " + outFile);
                } catch (Exception ex) {
                    System.out.println("Error writing file: " + ex.getMessage());
                }
                return;
            }
            Map<String, Object> arr = ExecutorImpl.arrayStore.get(arrName);
            if (arr == null || arr.isEmpty()) {
                System.out.println("Array not found or empty: " + arrName);
                return;
            }

            // Detect dimensionality from key format
            boolean is1D = true;
            int maxRow = 0, maxCol = 0;
            for (String key : arr.keySet()) {
                String[] dims = key.split("_");
                if (dims.length >= 2) {
                    is1D = false;
                    maxRow = Math.max(maxRow, Integer.parseInt(dims[0]));
                    maxCol = Math.max(maxCol, Integer.parseInt(dims[1]));
                } else {
                    maxRow = Math.max(maxRow, Integer.parseInt(dims[0]));
                }
            }

            StringBuilder csv = new StringBuilder();
            if (is1D) {
                for (int i = 0; i <= maxRow; i++) {
                    if (i > 0) csv.append(',');
                    Object v = arr.get(String.valueOf(i));
                    csv.append(v != null ? v : "0.0");
                }
                csv.append('\n');
            } else {
                for (int r = 0; r <= maxRow; r++) {
                    for (int c = 0; c <= maxCol; c++) {
                        if (c > 0) csv.append(',');
                        Object v = arr.get(r + "_" + c);
                        csv.append(v != null ? v : "0.0");
                    }
                    csv.append('\n');
                }
            }

            String dims = is1D ? String.valueOf(maxRow + 1) : (maxRow + 1) + "x" + (maxCol + 1);
            if (outFile != null) {
                try {
                    Files.write(Paths.get(outFile),
                            csv.toString().getBytes(StandardCharsets.UTF_8));
                    System.out.println("Dumped " + arrName + " (" + dims + ") to " + outFile);
                } catch (Exception e) {
                    System.out.println("Error writing file: " + e.getMessage());
                }
            } else {
                System.out.print(csv.toString());
            }
            return;
        }

        System.out.println("Unknown command: " + line);
    }

    static void takeBinaryArray(String sourcePath, String binaryOutPath) throws IOException {
        Path source = Paths.get(sourcePath);
        Path destination = Paths.get(binaryOutPath);
        try {
            Files.move(source, destination, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        } catch (java.nio.file.AtomicMoveNotSupportedException ex) {
            Files.copy(source, destination, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            Files.delete(source);
        } catch (java.nio.file.FileSystemException ex) {
            Files.copy(source, destination, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            Files.delete(source);
        }
    }

    private static void dumpBinaryArray(String sourcePath, String csvOutPath) throws IOException {
        Path source = Paths.get(sourcePath);
        Path csvOut = Paths.get(csvOutPath);
        String fileName = csvOut.getFileName().toString();
        String binName = fileName.endsWith(".csv")
                ? fileName.substring(0, fileName.length() - 4) + ".bin"
                : fileName + ".bin";
        Path binOut = csvOut.resolveSibling(binName);
        Files.copy(source, binOut, java.nio.file.StandardCopyOption.REPLACE_EXISTING);

        long floatCount = Files.size(binOut) / Float.BYTES;
        StringBuilder stub = new StringBuilder();
        int columns = floatCount > 1 && floatCount % 3072 == 0 ? 3072 : (int) floatCount;
        for (int i = 0; i < columns; i++) {
            if (i > 0) stub.append(',');
            stub.append('0');
        }
        stub.append('\n');
        if (floatCount > columns) stub.append("0\n");
        Files.write(csvOut, stub.toString().getBytes(StandardCharsets.UTF_8));
        Files.setLastModifiedTime(binOut, java.nio.file.attribute.FileTime.fromMillis(
                System.currentTimeMillis() + 2000
        ));
    }
}

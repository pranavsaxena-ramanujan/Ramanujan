package in.ramanujan.translation;

import in.ramanujan.developer.console.model.pojo.csv.CsvInformation;
import in.ramanujan.pojo.ruleEngineInputUnitsExt.array.Array;
import in.ramanujan.translation.codeConverter.utils.TranslateUtil;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

public class InlineCsvInformationTest {
    @Test
    public void inlineDataFlagIsInternalAndLargeInlineCsvUsesGeneratedWorkingDirectoryFile() throws Exception {
        com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
        CsvInformation csv = mapper.readValue("{\"fileName\":\"values.csv\",\"data\":\"1\",\"inlineData\":true}",
                CsvInformation.class);
        assertFalse(csv.isInlineData());
        csv.setInlineData(true);
        assertFalse(mapper.writeValueAsString(csv).contains("inlineData"));
        StringBuilder data = new StringBuilder(900000);
        for (int i = 0; i < 200001; i++) data.append("1.0,");
        data.append("1.0\n");
        csv.setData(data.toString());
        Array array = new Array();
        array.setName("values");
        Map<String, Array> arrays = new HashMap<>();
        arrays.put("id_name_values", array);
        try {
            new TranslateUtil().repopulateCsvArrayValues(arrays, Collections.singletonList(csv));
            assertNotNull(array.getBinaryFile());
            Path binary = Paths.get(array.getBinaryFile());
            assertEquals(Paths.get(".").toFile().getCanonicalFile(), binary.toFile().getParentFile().getCanonicalFile());
            assertEquals(200002L * Float.BYTES, Files.size(binary));
        } finally {
            if (array.getBinaryFile() != null) Files.deleteIfExists(Paths.get(array.getBinaryFile()));
        }
    }

    @Test
    public void inlineCsvCannotLoadExistingServerSidecarButLegacyCsvStillCan() throws Exception {
        Path binary = Files.createTempFile(Paths.get("."), "inline_csv_guard_", ".bin");
        try {
            Files.write(binary, new byte[]{0, 0, 0, 0});
            String csvPath = binary.toString().replaceFirst("\\.bin$", ".csv");
            String name = binary.getFileName().toString().replaceFirst("\\.bin$", "");
            CsvInformation csv = new CsvInformation();
            csv.setFileName(csvPath);
            csv.setData("9.0\n");
            csv.setInlineData(true);
            Array array = new Array();
            array.setName(name);
            Map<String, Array> arrays = new HashMap<>();
            arrays.put("id_name_" + name, array);
            TranslateUtil translate = new TranslateUtil();
            translate.repopulateCsvArrayValues(arrays, Collections.singletonList(csv));
            assertNull(array.getBinaryFile());
            assertEquals(9.0, ((Number) array.getValues().get("0")).doubleValue(), 0.0);
            csv.setInlineData(false);
            translate.repopulateCsvArrayValues(arrays, Collections.singletonList(csv));
            assertEquals(binary.toFile().getCanonicalPath(), array.getBinaryFile());
        } finally {
            Files.deleteIfExists(binary);
        }
    }
}

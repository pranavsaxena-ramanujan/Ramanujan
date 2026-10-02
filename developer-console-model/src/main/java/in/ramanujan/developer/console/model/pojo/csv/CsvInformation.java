package in.ramanujan.developer.console.model.pojo.csv;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.Data;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class CsvInformation {
    private String fileName;
    private String data;
    @JsonIgnore
    private boolean inlineData;
}

package in.ramanujan.db.layer;

import in.ramanujan.db.layer.constants.Keys;
import in.ramanujan.db.layer.enums.QueryType;
import in.ramanujan.db.layer.schema.VariableMapping;
import in.ramanujan.db.layer.utils.InMemQueryExecutor;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class InMemSelectInTest {
    private static VariableMapping mapping(String asyncId, String variableId) {
        VariableMapping mapping = new VariableMapping();
        mapping.setAsyncId(asyncId);
        mapping.setVariableId(variableId);
        mapping.setObject(variableId + "-value");
        return mapping;
    }

    @Test
    public void selectInReturnsEachMatchingRowOnceForTheGivenAsyncId() throws Exception {
        InMemQueryExecutor executor = new InMemQueryExecutor();
        executor.init(null);
        for (VariableMapping row : Arrays.asList(mapping("a1", "v1"), mapping("a1", "v2"), mapping("a1", "v3"), mapping("a2", "v1"))) {
            executor.execute(row, null, QueryType.INSERT);
        }
        VariableMapping template = new VariableMapping();
        template.setAsyncId("a1");
        List<Object> batch = Collections.singletonList(new ArrayList<Object>(Arrays.asList("v1", "v3", "missing")));
        List<Object> result = executor.execute(template, Keys.VARIABLE_IDS_IN, QueryType.SELECT_IN, batch).result();
        Assert.assertEquals(2, result.size());
        List<String> ids = new ArrayList<>();
        for (Object row : result) {
            Assert.assertEquals("a1", ((VariableMapping) row).getAsyncId());
            ids.add(((VariableMapping) row).getVariableId());
        }
        Collections.sort(ids);
        Assert.assertEquals(Arrays.asList("v1", "v3"), ids);
    }
}

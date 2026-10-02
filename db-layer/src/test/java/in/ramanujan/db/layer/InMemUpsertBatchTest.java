package in.ramanujan.db.layer;

import in.ramanujan.db.layer.constants.Keys;
import in.ramanujan.db.layer.enums.QueryType;
import in.ramanujan.db.layer.schema.ArrayMapping;
import in.ramanujan.db.layer.schema.VariableMapping;
import in.ramanujan.db.layer.utils.InMemQueryExecutor;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class InMemUpsertBatchTest {
    private static ArrayMapping cell(String index, String value) {
        ArrayMapping mapping = new ArrayMapping();
        mapping.setAsyncId("a1");
        mapping.setArrayId("arr");
        mapping.setArrayName("");
        mapping.setIndexStr(index);
        mapping.setObject(value);
        return mapping;
    }

    private static List<Object> batch(String... indexValuePairs) {
        List<Object> rows = new ArrayList<>();
        for (int i = 0; i < indexValuePairs.length; i += 2) rows.add(cell(indexValuePairs[i], indexValuePairs[i + 1]));
        return rows;
    }

    private static List<Object> select(InMemQueryExecutor executor, String index) throws Exception {
        return executor.execute(cell(index, null), Keys.ASYNC_ID_ARRAY_ID_INDEX, QueryType.SELECT).result();
    }

    @Test
    public void batchUpsertStoresEveryRowAndUpdatesExistingOnes() throws Exception {
        InMemQueryExecutor executor = new InMemQueryExecutor();
        executor.init(null);
        List<Object> first = batch("0", "x", "1", "y");
        Assert.assertTrue(executor.execute(first.get(0), Keys.ASYNC_ID_ARRAY_ID_INDEX, QueryType.UPSERT, first).succeeded());
        List<Object> second = batch("1", "z", "2", "w");
        Assert.assertTrue(executor.execute(second.get(0), Keys.ASYNC_ID_ARRAY_ID_INDEX, QueryType.UPSERT, second).succeeded());

        Assert.assertEquals("x", ((ArrayMapping) select(executor, "0").get(0)).getObject());
        List<Object> updated = select(executor, "1");
        Assert.assertEquals(1, updated.size());
        Assert.assertEquals("z", ((ArrayMapping) updated.get(0)).getObject());
        Assert.assertEquals("w", ((ArrayMapping) select(executor, "2").get(0)).getObject());
    }

    @Test
    public void batchInsertAndUpdateApplyToEveryRow() throws Exception {
        InMemQueryExecutor executor = new InMemQueryExecutor();
        executor.init(null);
        List<Object> inserts = new ArrayList<>();
        for (String id : new String[]{"v1", "v2", "v3"}) {
            VariableMapping row = new VariableMapping();
            row.setAsyncId("a1");
            row.setVariableId(id);
            row.setObject("0");
            inserts.add(row);
        }
        Assert.assertTrue(executor.execute(inserts.get(0), Keys.ASYNC_ID_VARIABLE_ID, QueryType.INSERT, inserts).succeeded());
        List<Object> updates = new ArrayList<>();
        for (String id : new String[]{"v1", "v2", "v3"}) {
            VariableMapping update = new VariableMapping();
            update.setVariableId(id);
            update.setObject(id + "-new");
            updates.add(update);
        }
        Assert.assertTrue(executor.execute(updates.get(0), Keys.VARIABLE_ID, QueryType.UPDATE, updates).succeeded());

        VariableMapping template = new VariableMapping();
        template.setAsyncId("a1");
        List<Object> rows = executor.execute(template, Keys.ASYNC_ID, QueryType.SELECT).result();
        Assert.assertEquals(3, rows.size());
        for (Object row : rows) {
            VariableMapping mapping = (VariableMapping) row;
            Assert.assertEquals(mapping.getVariableId() + "-new", mapping.getObject());
        }
    }
}

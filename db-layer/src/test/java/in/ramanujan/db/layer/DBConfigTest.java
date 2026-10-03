package in.ramanujan.db.layer;

import in.ramanujan.db.layer.utils.QueryExecutor;
import org.junit.Assert;
import org.junit.Test;

public class DBConfigTest {
    @Test
    public void configurationLogsDoNotContainPasswords() {
        QueryExecutor.DBConfig config = new QueryExecutor.DBConfig(
                "localhost:3306", "test-user", "test-only-password", "test-db");
        Assert.assertFalse(config.toString().contains("test-only-password"));
        Assert.assertTrue(config.toString().contains("test-db"));
    }
}

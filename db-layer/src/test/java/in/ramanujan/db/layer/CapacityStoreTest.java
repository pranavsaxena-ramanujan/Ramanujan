package in.ramanujan.db.layer;

import in.ramanujan.db.layer.utils.CapacityStore;
import in.ramanujan.db.layer.utils.QueryExecutor;
import io.vertx.core.Future;
import io.vertx.core.Vertx;
import org.h2.jdbcx.JdbcDataSource;
import org.junit.Test;

import java.io.*;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.sql.*;
import java.util.*;
import java.util.concurrent.*;

import static org.junit.Assert.*;

public class CapacityStoreTest {
    private static void inject(Object object, String name, Object value) throws Exception {
        Field field = object.getClass().getDeclaredField(name);
        field.setAccessible(true);
        field.set(object, value);
    }

    private static <T> T await(Future<T> future) throws Exception {
        CompletableFuture<T> value = new CompletableFuture<>();
        future.setHandler(done -> {
            if (done.succeeded()) value.complete(done.result()); else value.completeExceptionally(done.cause());
        });
        return value.get(10, TimeUnit.SECONDS);
    }

    @Test
    public void sqlAdmissionSerializesAcrossStoreInstancesAndRollsBack() throws Exception {
        Vertx vertx = Vertx.vertx();
        try {
            JdbcDataSource source = new JdbcDataSource();
            source.setURL("jdbc:h2:mem:capacity_" + UUID.randomUUID() + ";MODE=MySQL;DB_CLOSE_DELAY=-1;LOCK_TIMEOUT=5000");
            try (InputStream input = getClass().getResourceAsStream("/migrations/20261002_capacity_admission.sql");
                 ByteArrayOutputStream bytes = new ByteArrayOutputStream();
                 Connection connection = source.getConnection();
                 Statement statement = connection.createStatement()) {
                assertNotNull(input);
                byte[] buffer = new byte[4096];
                int count;
                while ((count = input.read(buffer)) >= 0) bytes.write(buffer, 0, count);
                String sql = new String(bytes.toByteArray(), StandardCharsets.UTF_8);
                for (String command : sql.split(";")) if (!command.trim().isEmpty()) statement.execute(command);
            }
            QueryExecutor query = new QueryExecutor();
            inject(query, "dbType", QueryExecutor.DB_TYPE.GCP);
            inject(query, "context", vertx.getOrCreateContext());
            inject(query, "dataSource", source);
            CapacityStore first = new CapacityStore();
            CapacityStore second = new CapacityStore();
            inject(first, "queryExecutor", query);
            inject(second, "queryExecutor", query);
            await(first.atomic("room", state -> {
                Map<String, Object> row = new LinkedHashMap<>();
                row.put("counter", 0);
                row.put("state", "ACTIVE");
                state.plans.put("plan", row);
                return null;
            }));
            Future<Integer> one = first.atomic("room", state -> increment(state));
            Future<Integer> two = second.atomic("room", state -> increment(state));
            Set<Integer> values = new HashSet<>(Arrays.asList(await(one), await(two)));
            assertEquals(new HashSet<>(Arrays.asList(1, 2)), values);
            try {
                await(first.atomic("room", state -> {
                    increment(state);
                    throw new IllegalStateException("admission rejected");
                }));
                fail("Expected rollback");
            } catch (ExecutionException expected) { assertEquals("admission rejected", expected.getCause().getMessage()); }
            assertEquals(2, (int) await(second.atomic("room", state -> ((Number) state.plans.get("plan").get("counter")).intValue())));
            assertTrue(await(second.atomic("other-room", state -> state.plans.isEmpty())));
            try (Connection connection = source.getConnection(); Statement statement = connection.createStatement()) {
                statement.execute("SHUTDOWN");
            }
        } finally {
            CountDownLatch stopped = new CountDownLatch(1);
            vertx.close(done -> stopped.countDown());
            assertTrue(stopped.await(5, TimeUnit.SECONDS));
        }
    }

    private static int increment(CapacityStore.State state) {
        Map<String, Object> row = state.plans.get("plan");
        int value = ((Number) row.get("counter")).intValue() + 1;
        row.put("counter", value);
        return value;
    }
}

package in.ramanujan.data.db.impl.storageDao;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;

public class StorageDaoLocalContainerImpl extends StorageDaoInternal {
    private File file(String id, String bucket) {
        String root = System.getenv("RAMANUJAN_STORAGE_ROOT");
        return new File(root == null || root.isEmpty() ? "/" : root, bucket + "/" + id);
    }

    /**
    * save a file in /buckName/objectId
    */
    @Override
    protected void setObject(String objectId, String buckName, String object, int currentRetryCount) throws Exception {
        try {
            File file = file(objectId, buckName);
            // write object to file
            Files.createDirectories(file.toPath().getParent());
            Files.write(file.toPath(), object.getBytes(StandardCharsets.UTF_8));
        } catch (Exception ex) {
            throw ex;
        }
    }

    @Override
    protected String getObject(String objectId, String bucketName, int currentRetryCount) throws Exception {
        try {
            File file = file(objectId, bucketName);
            return new String(Files.readAllBytes(file.toPath()));
        } catch (NoSuchFileException ex) {
            return "";
        }
    }
}

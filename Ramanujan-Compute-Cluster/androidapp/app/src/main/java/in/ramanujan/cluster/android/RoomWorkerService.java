package in.ramanujan.cluster.android;

import android.app.*;
import android.content.*;
import android.content.pm.ServiceInfo;
import android.os.*;
import in.ramanujan.cluster.client.common.JoinClient;
import in.ramanujan.devices.common.WorkerCapacity;
import in.ramanujan.developer.console.operationImpl.ExecuteInlineWorker;
import java.util.Arrays;
import java.util.concurrent.*;

public final class RoomWorkerService extends Service {
    private static final String CHANNEL = "cluster_worker";
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private volatile ExecuteInlineWorker worker;
    private boolean started;
    private volatile boolean stopped;
    private SharedPreferences prefs;

    @Override public void onCreate() {
        super.onCreate();
        prefs = getSharedPreferences(RoomActivity.PREFS, MODE_PRIVATE);
        NotificationManager manager = getSystemService(NotificationManager.class);
        manager.createNotificationChannel(new NotificationChannel(CHANNEL, "Room worker", NotificationManager.IMPORTANCE_LOW));
    }
    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && "stop".equals(intent.getAction())) { stopSelf(); return START_NOT_STICKY; }
        if (started) return START_NOT_STICKY;
        PendingIntent open = PendingIntent.getActivity(this, 0, new Intent(this, RoomActivity.class), PendingIntent.FLAG_IMMUTABLE);
        PendingIntent stop = PendingIntent.getService(this, 1, new Intent(this, RoomWorkerService.class).setAction("stop"), PendingIntent.FLAG_IMMUTABLE);
        Notification notification = new Notification.Builder(this, CHANNEL).setContentTitle("Ramanujan room worker")
                .setContentText("Starting; tap Stop to disconnect").setSmallIcon(android.R.drawable.stat_notify_sync)
                .setContentIntent(open).setOngoing(true).addAction(android.R.drawable.ic_media_pause, "Stop", stop).build();
        if (Build.VERSION.SDK_INT >= 29) startForeground(1, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
        else startForeground(1, notification);
        started = true;
        executor.submit(() -> {
            try {
                String workerUrl = prefs.getString("workerUrl", "");
                JoinClient.validateTransport(prefs.getString("portal", ""), prefs.getBoolean("allowLocalHttp", false));
                JoinClient.validateWorkerUrl(prefs.getString("portal", ""), workerUrl);
                prefs.edit().putString("status", "Checking native libraries and OpenCL…").apply();
                NativePreflight.verify(this);
                if (stopped) return;
                ExecuteInlineWorker running = new ExecuteInlineWorker(root -> {
                    ActivityManager.MemoryInfo memory = new ActivityManager.MemoryInfo();
                    getSystemService(ActivityManager.class).getMemoryInfo(memory);
                    Long disk = null;
                    try { disk = new StatFs(root.toString()).getAvailableBytes(); }
                    catch (IllegalArgumentException ignored) {}
                    return new WorkerCapacity.Snapshot(memory.totalMem, memory.availMem, disk);
                });
                worker = running;
                if (stopped) { running.stop(); return; }
                prefs.edit().putString("status", "Running — private room, OpenCL available (GPU compatibility untested)").apply();
                running.execute(Arrays.asList("worker", workerUrl, "1", "--cache",
                        new java.io.File(getFilesDir(), "localcache").getAbsolutePath(), "--llm-sessions", "8"));
            } catch (Exception | LinkageError e) {
                String message = e.getMessage();
                String failure = message != null && message.startsWith("OpenCL unavailable:")
                        ? message : "Worker unavailable: verify native libraries, OpenCL support and the saved room.";
                prefs.edit().putString("status", failure).apply();
            } finally { stopSelf(); }
        });
        return START_NOT_STICKY;
    }
    @Override public void onDestroy() {
        stopped = true;
        ExecuteInlineWorker running = worker;
        if (running != null) running.stop();
        executor.shutdownNow();
        String previous = prefs.getString("status", "");
        if (previous.startsWith("Running") || previous.startsWith("Checking")) prefs.edit().putString("status", "Stopped").apply();
        stopForeground(STOP_FOREGROUND_REMOVE);
        super.onDestroy();
    }
    @Override public IBinder onBind(Intent intent) { return null; }
}

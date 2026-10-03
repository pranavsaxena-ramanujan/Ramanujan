package in.ramanujan.cluster.android;

import android.Manifest;
import android.app.Activity;
import android.content.*;
import android.os.*;
import android.text.InputType;
import android.view.View;
import android.widget.*;
import in.ramanujan.cluster.client.common.*;
import java.util.concurrent.*;

public final class RoomActivity extends Activity {
    static final String PREFS = "private_room";
    private EditText portal, room, secret, name;
    private TextView status;
    private Button join, start;
    private CheckBox localHttp;
    private final ExecutorService network = Executors.newSingleThreadExecutor();
    private final Handler handler = new Handler(Looper.getMainLooper());
    private SharedPreferences prefs;
    private final Runnable refresh = new Runnable() {
        public void run() {
            if (status != null) status.setText(prefs.getString("status", "Stopped"));
            handler.postDelayed(this, 1000);
        }
    };

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL); content.setPadding(24, 24, 24, 24);
        TextView title = new TextView(this); title.setText("Join a private Ramanujan room"); content.addView(title);
        portal = field(content, "Portal URL", prefs.getString("portal", "https://"));
        room = field(content, "Room ID", prefs.getString("roomId", ""));
        secret = field(content, "Join secret", "");
        secret.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD);
        name = field(content, "Device name", prefs.getString("name", Build.MODEL));
        localHttp = new CheckBox(this);
        localHttp.setText("Allow localhost/private-IP HTTP (development only)");
        localHttp.setChecked(prefs.getBoolean("allowLocalHttp", false)); content.addView(localHttp);
        join = button(content, "Join room", v -> join());
        start = button(content, "Start worker", v -> start());
        button(content, "Stop worker", v -> {
            stopService(new Intent(this, RoomWorkerService.class));
            prefs.edit().putString("status", "Stopped").apply();
        });
        status = new TextView(this); content.addView(status);
        ScrollView scroll = new ScrollView(this); scroll.addView(content); setContentView(scroll);
        start.setEnabled(prefs.contains("workerUrl"));
        if (Build.VERSION.SDK_INT >= 33) requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 1);
    }

    private EditText field(LinearLayout parent, String hint, String value) {
        EditText field = new EditText(this); field.setHint(hint); field.setText(value);
        field.setSingleLine(true); parent.addView(field); return field;
    }
    private Button button(LinearLayout parent, String text, View.OnClickListener action) {
        Button button = new Button(this); button.setText(text); button.setOnClickListener(action); parent.addView(button); return button;
    }

    private void join() {
        try { JoinClient.validateTransport(portal.getText().toString(), localHttp.isChecked()); }
        catch (IllegalArgumentException e) {
            prefs.edit().putString("status", e.getMessage()).apply(); status.setText(e.getMessage()); return;
        }
        stopService(new Intent(this, RoomWorkerService.class));
        String url = portal.getText().toString(), id = room.getText().toString(), deviceName = name.getText().toString();
        String password = secret.getText().toString(); secret.setText("");
        boolean allowLocalHttp = localHttp.isChecked();
        join.setEnabled(false); start.setEnabled(false); status.setText("Joining…");
        network.submit(() -> {
            boolean joined = false;
            try {
                Registration registration = new JoinClient().join(url, id, password, deviceName, "android", allowLocalHttp);
                prefs.edit().putString("portal", JoinClient.portal(url)).putString("roomId", id)
                        .putString("name", deviceName).putString("deviceId", registration.deviceId)
                        .putString("clusterId", registration.clusterId).putString("workerUrl", registration.workerUrl)
                        .putBoolean("allowLocalHttp", allowLocalHttp).putString("status", "Joined — ready to reconnect").commit();
                joined = true;
            } catch (Exception e) { prefs.edit().putString("status", "Join failed. Check the portal, room and secret.").apply(); }
            final boolean success = joined;
            runOnUiThread(() -> { join.setEnabled(true); start.setEnabled(success || prefs.contains("workerUrl")); });
        });
    }

    private void start() {
        if (!prefs.contains("workerUrl")) return;
        startForegroundService(new Intent(this, RoomWorkerService.class));
    }
    @Override protected void onResume() { super.onResume(); handler.post(refresh); }
    @Override protected void onPause() { handler.removeCallbacks(refresh); super.onPause(); }
    @Override protected void onDestroy() { network.shutdownNow(); super.onDestroy(); }
}

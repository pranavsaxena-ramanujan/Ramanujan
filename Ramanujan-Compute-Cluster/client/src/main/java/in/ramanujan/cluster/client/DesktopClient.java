package in.ramanujan.cluster.client;

import in.ramanujan.cluster.client.common.*;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;

public final class DesktopClient extends JFrame {
    private final JTextField portal = new JTextField(), room = new JTextField(), name = new JTextField();
    private final JPasswordField secret = new JPasswordField();
    private final JCheckBox localHttp = new JCheckBox("Allow local HTTP (development only)");
    private final JTextArea logs = new JTextArea(16, 70);
    private final JLabel status = new JLabel("Stopped");
    private final JButton join = new JButton("Join room"), start = new JButton("Start worker"), stop = new JButton("Stop worker");
    private final Path home = Paths.get(System.getProperty("user.home"), ".ramanujan", "cluster-client");
    private final ConfigStore config = new ConfigStore(home);
    private Registration registration;
    private volatile Process worker;

    private DesktopClient() {
        super("Ramanujan Compute Cluster");
        JPanel form = new JPanel(new GridLayout(0, 2, 6, 6));
        form.add(new JLabel("Portal URL (use HTTPS)")); form.add(portal);
        form.add(new JLabel("Room ID")); form.add(room);
        form.add(new JLabel("Join secret")); form.add(secret);
        form.add(new JLabel("Device name")); form.add(name);
        form.add(new JLabel("HTTPS required for production")); form.add(localHttp);
        form.add(join); form.add(start); form.add(status); form.add(stop);
        logs.setEditable(false); logs.setLineWrap(true);
        add(form, BorderLayout.NORTH); add(new JScrollPane(logs), BorderLayout.CENTER);
        start.setEnabled(false); stop.setEnabled(false);
        try {
            Properties saved = config.load();
            portal.setText(saved.getProperty("portal", "http://localhost:8090")); room.setText(saved.getProperty("roomId", ""));
            name.setText(saved.getProperty("name", System.getProperty("user.name") + " desktop"));
            localHttp.setSelected(Boolean.parseBoolean(saved.getProperty("allowLocalHttp", "false")));
            if (saved.containsKey("workerUrl")) {
                registration = new Registration(saved.getProperty("deviceId"), saved.getProperty("clusterId"), saved.getProperty("workerUrl"));
                start.setEnabled(true); status.setText("Room saved — ready to reconnect");
            }
        } catch (IOException e) { log("Saved room could not be loaded. Join again."); }
        join.addActionListener(e -> joinRoom()); start.addActionListener(e -> startWorker()); stop.addActionListener(e -> stopWorker());
        setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE);
        addWindowListener(new WindowAdapter() { public void windowClosing(WindowEvent e) { stopWorker(); dispose(); } });
        pack(); setLocationRelativeTo(null);
    }

    private void joinRoom() {
        try { JoinClient.validateTransport(portal.getText(), localHttp.isSelected()); }
        catch (IllegalArgumentException e) { status.setText("Invalid portal"); log(e.getMessage()); return; }
        join.setEnabled(false); start.setEnabled(false);
        String url = portal.getText(), roomId = room.getText(), deviceName = name.getText();
        char[] password = secret.getPassword(); secret.setText("");
        boolean allowLocalHttp = localHttp.isSelected();
        status.setText("Joining…");
        new SwingWorker<Registration, Void>() {
            protected Registration doInBackground() throws Exception {
                try {
                    Registration joined = new JoinClient().join(url, roomId, new String(password), deviceName,
                            platform(), allowLocalHttp);
                    config.save(JoinClient.portal(url), roomId, deviceName, joined, allowLocalHttp);
                    return joined;
                } finally { Arrays.fill(password, '\0'); }
            }
            protected void done() {
                try { registration = get(); status.setText("Joined — ready"); log("Device registered in the private room."); }
                catch (Exception e) { status.setText("Join failed"); log("Join failed: check URL, room and secret, then retry."); }
                join.setEnabled(worker == null); start.setEnabled(worker == null && registration != null);
            }
        }.execute();
    }

    private void startWorker() {
        if (registration == null || worker != null) return;
        try {
            Path bundle = Paths.get(DesktopClient.class.getProtectionDomain().getCodeSource().getLocation().toURI()).getParent();
            Process process = WorkerLauncher.command(bundle, home.resolve("localcache"), registration.workerUrl).start();
            worker = process; start.setEnabled(false); join.setEnabled(false); stop.setEnabled(true); status.setText("Starting…");
            Thread reader = new Thread(() -> {
                try (BufferedReader input = new BufferedReader(new InputStreamReader(process.getInputStream(), "UTF-8"))) {
                    String line;
                    while ((line = input.readLine()) != null) {
                        String safe = JoinClient.redact(line).replace(registration.workerUrl, "[private room gateway]");
                        SwingUtilities.invokeLater(() -> {
                            log(safe);
                            if ("LOCAL_WORKER_READY".equals(safe)) status.setText("Running");
                        });
                    }
                    int exit = process.waitFor();
                    SwingUtilities.invokeLater(() -> {
                        if (worker == process) worker = null;
                        status.setText("Stopped (exit " + exit + ")");
                        start.setEnabled(registration != null); join.setEnabled(true); stop.setEnabled(false);
                    });
                } catch (IOException | InterruptedException e) { SwingUtilities.invokeLater(() -> log("Worker output disconnected.")); }
            }, "worker-log-reader");
            reader.setDaemon(true); reader.start();
        } catch (Exception e) { status.setText("Start failed"); log("Worker could not start. Verify the installed runtime, worker JAR and both native libraries."); }
    }

    private void stopWorker() {
        Process process = worker;
        if (process == null) return;
        status.setText("Stopping…"); stop.setEnabled(false);
        process.destroy();
        Thread terminator = new Thread(() -> {
            try { if (!process.waitFor(5, java.util.concurrent.TimeUnit.SECONDS)) process.destroyForcibly(); }
            catch (InterruptedException e) { Thread.currentThread().interrupt(); }
        }, "worker-stop");
        terminator.start();
    }

    private void log(String line) {
        logs.append(line + "\n");
        if (logs.getDocument().getLength() > 100000) logs.replaceRange("", 0, 50000);
        logs.setCaretPosition(logs.getDocument().getLength());
    }

    private static String platform() {
        String os = System.getProperty("os.name").toLowerCase(Locale.ROOT);
        return os.contains("win") ? "windows" : os.contains("mac") ? "macos" : "linux";
    }

    public static void main(String[] args) { SwingUtilities.invokeLater(() -> new DesktopClient().setVisible(true)); }
}

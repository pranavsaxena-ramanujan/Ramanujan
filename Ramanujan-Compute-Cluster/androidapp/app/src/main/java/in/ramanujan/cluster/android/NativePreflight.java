package in.ramanujan.cluster.android;

import android.content.Context;
import android.system.Os;
import in.ramanujan.rule.engine.LlmSession;

final class NativePreflight {
    static void verify(Context context) throws Exception {
        Os.setenv("TMPDIR", context.getCacheDir().getAbsolutePath(), true);
        System.loadLibrary("native");
        System.loadLibrary("cluster_preflight");
        String unavailable = checkOpenCl();
        if (unavailable != null) throw new IllegalStateException(unavailable);
        LlmSession.ensureLoaded();
    }
    private static native String checkOpenCl();
}

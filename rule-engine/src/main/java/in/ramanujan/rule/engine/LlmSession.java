package in.ramanujan.rule.engine;

import in.ramanujan.rule.engine.util.NativeLibraryLoader;

import java.io.IOException;
import java.util.Locale;

/**
 * One pipeline stage of a decoder-only LLM running on the native OpenCL runtime
 * (libramanujan_llm, sources in ramanujan-native/native/llm).
 *
 * The stage graph JSON (sharded-llm/converter/ramanujan_shards/llm_graph.py) names the
 * weight files, their GGUF encodings and the layer structure. The session owns the device
 * weights, KV caches and recurrent state, and advances them token by token. Calls on one
 * session are serialized; separate sessions may run concurrently.
 */
public final class LlmSession implements AutoCloseable {
    private static final Object LOAD_LOCK = new Object();
    private static volatile boolean loaded;

    private long handle;

    public LlmSession(String graphJson) {
        ensureLoaded();
        handle = nativeOpen(graphJson);
    }

    /** Loads libramanujan_llm (from $RAMANUJAN_WS on desktop JVMs, from the APK on Android). */
    public static void ensureLoaded() {
        if (loaded) return;
        synchronized (LOAD_LOCK) {
            if (loaded) return;
            String name = System.getProperty("ramanujan.llmLibrary", "ramanujan_llm");
            String runtime = System.getProperty("java.runtime.name", "");
            if (runtime.toLowerCase(Locale.ROOT).contains("android")) {
                System.loadLibrary(name);
            } else {
                try {
                    NativeLibraryLoader.load(name);
                } catch (IOException e) {
                    throw new IllegalStateException("cannot load native LLM runtime: " + e.getMessage(), e);
                }
            }
            loaded = true;
        }
    }

    /**
     * Advances the stage by {@code n} tokens starting at {@code pos}, which must equal the
     * number of tokens already consumed. Pass token ids when the stage has the embedding,
     * otherwise {@code n * dim} hidden values. Returns the last token's logits when the
     * stage has the output head, otherwise {@code n * dim} hidden values.
     */
    public synchronized float[] step(int[] tokens, float[] hidden, int n, int pos) {
        return nativeStep(requireOpen(), tokens, hidden, n, pos);
    }

    /** Clears KV caches and recurrent state; the next step starts at position 0. */
    public synchronized void reset() {
        nativeReset(requireOpen());
    }

    /** JSON: device, weights mode, bytes, position and timings. */
    public synchronized String info() {
        return nativeInfo(requireOpen());
    }

    /** Already-selected device telemetry, or null before library load. Does not initialize OpenCL. */
    public static String capacityInfo() {
        return loaded ? nativeCapacityInfo() : null;
    }

    public static boolean isLoaded() { return loaded; }

    /** Selects the runtime device and prepares kernels, without opening or resetting any session. */
    public static void prepareCapacity() {
        ensureLoaded();
        nativePrepareCapacity();
    }

    @Override
    public synchronized void close() {
        if (handle != 0) {
            nativeClose(handle);
            handle = 0;
        }
    }

    private long requireOpen() {
        if (handle == 0) throw new IllegalStateException("LLM session is closed");
        return handle;
    }

    private static native long nativeOpen(String graphJson);

    private static native float[] nativeStep(long handle, int[] tokens, float[] hidden, int n, int pos);

    private static native void nativeReset(long handle);

    private static native String nativeInfo(long handle);

    private static native String nativeCapacityInfo();

    private static native void nativePrepareCapacity();

    private static native void nativeClose(long handle);
}

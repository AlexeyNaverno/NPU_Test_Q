package ru.npu.llamachat.engine

/**
 * DTO-классы и низкоуровневый JNI-мост к нативной библиотеке libnpu_llama_jni.so.
 * Сигнатуры конструкторов строго синхронизированы с app/src/main/cpp/npu_llama_jni.cpp.
 */
data class NativeInitResult(
    val ok: Boolean,
    val hexagonArch: Int,
    val htpDeviceCount: Int,
    val hasV73Skel: Boolean,
    val activeDevices: String,
    val hwSummary: String,
    val errorMessage: String
)

data class NativeModelLoadResult(
    val ok: Boolean,
    val modelDesc: String,
    val quantName: String,
    val modelBytes: Long,
    val modelParams: Long,
    val nCtxTrain: Int,
    val nCtxActive: Int,
    val nLayersTotal: Int,
    val nLayersNpu: Int,
    val htpBufferMib: Double,
    val hostBufferMib: Double,
    val npuDeviceUsed: String,
    val errorMessage: String
)

data class NativeGenerationStats(
    val ok: Boolean,
    val stoppedByUser: Boolean,
    val promptTokens: Int,
    val cachedTokens: Int,
    val generatedTokens: Int,
    val contextUsedTokens: Int,
    val contextMaxTokens: Int,
    val prefillTokPerSec: Double,
    val decodeTokPerSec: Double,
    val errorMessage: String
)

data class NativeMemoryInfo(
    val processRssMib: Double,
    val processPeakRssMib: Double,
    val processSwapMib: Double,
    val systemTotalMib: Double,
    val systemAvailMib: Double,
    val npuHtpBufferMib: Double,
    val hostWeightMib: Double,
    val contextUsedTokens: Int,
    val contextMaxTokens: Int
)

object NpuLlamaBridge {

    fun interface ProgressListener {
        fun onProgress(progress: Float)
    }

    fun interface TokenStreamListener {
        /** Возвращает true для продолжения генерации или false для немедленной остановки. */
        fun onTokenDelta(piece: String): Boolean
    }

    @Volatile
    private var librariesLoaded = false

    @Volatile
    var libraryLoadError: String? = null
        private set

    @Synchronized
    fun ensureLibrariesLoaded(): Boolean {
        if (librariesLoaded) return true
        return try {
            System.loadLibrary("ggml-base")
            runCatching { System.loadLibrary("ggml-cpu") }
            // В официальном тарболе llama.cpp Snapdragon libggml.so может иметь DT_NEEDED на libggml-opencl.so;
            // загружаем опционально, при этом в C++ выполнение на GPU полностью заблокировано (только HTP0!).
            runCatching { System.loadLibrary("ggml-opencl") }
            System.loadLibrary("ggml-hexagon")
            System.loadLibrary("ggml")
            System.loadLibrary("llama")
            System.loadLibrary("npu_llama_jni")
            librariesLoaded = true
            libraryLoadError = null
            true
        } catch (t: Throwable) {
            libraryLoadError = "Ошибка загрузки нативных библиотек Qualcomm Hexagon NPU: ${t.message}"
            false
        }
    }

    external fun nativeInitNpu(
        nativeLibDir: String,
        htpDevices: String,
        powerMode: String
    ): NativeInitResult

    external fun nativeLoadModelFromFd(
        detachedFd: Int,
        displayName: String,
        nCtx: Int,
        maxNewTokens: Int,
        nBatch: Int,
        nThreads: Int,
        nNpuLayers: Int,
        temperature: Float,
        topP: Float,
        topK: Int,
        minP: Float,
        repeatPenalty: Float,
        repeatLastN: Int,
        seed: Int,
        flashAttnMode: Int,
        kvQ8: Boolean,
        powerMode: String,
        htpDevices: String,
        systemPrompt: String,
        progressListener: ProgressListener?
    ): NativeModelLoadResult

    external fun nativeLoadModelFromPath(
        filePath: String,
        nCtx: Int,
        maxNewTokens: Int,
        nBatch: Int,
        nThreads: Int,
        nNpuLayers: Int,
        temperature: Float,
        topP: Float,
        topK: Int,
        minP: Float,
        repeatPenalty: Float,
        repeatLastN: Int,
        seed: Int,
        flashAttnMode: Int,
        kvQ8: Boolean,
        powerMode: String,
        htpDevices: String,
        systemPrompt: String,
        progressListener: ProgressListener?
    ): NativeModelLoadResult

    external fun nativeApplyConfig(
        nCtx: Int,
        maxNewTokens: Int,
        nBatch: Int,
        nThreads: Int,
        nNpuLayers: Int,
        temperature: Float,
        topP: Float,
        topK: Int,
        minP: Float,
        repeatPenalty: Float,
        repeatLastN: Int,
        seed: Int,
        flashAttnMode: Int,
        kvQ8: Boolean,
        powerMode: String,
        htpDevices: String,
        systemPrompt: String
    ): String

    external fun nativeGenerate(
        roles: Array<String>,
        contents: Array<String>,
        nCtx: Int,
        maxNewTokens: Int,
        nBatch: Int,
        nThreads: Int,
        nNpuLayers: Int,
        temperature: Float,
        topP: Float,
        topK: Int,
        minP: Float,
        repeatPenalty: Float,
        repeatLastN: Int,
        seed: Int,
        flashAttnMode: Int,
        kvQ8: Boolean,
        powerMode: String,
        htpDevices: String,
        systemPrompt: String,
        tokenListener: TokenStreamListener?
    ): NativeGenerationStats

    external fun nativeCancelGeneration()
    external fun nativeClearChat()
    external fun nativeUnloadModel()
    external fun nativeCountTokens(text: String): Int
    external fun nativeGetMemoryInfo(): NativeMemoryInfo
}

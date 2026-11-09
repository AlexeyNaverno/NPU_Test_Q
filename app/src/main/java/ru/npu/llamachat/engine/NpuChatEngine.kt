package ru.npu.llamachat.engine

import android.app.ActivityManager
import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import ru.npu.llamachat.model.ChatMessage
import ru.npu.llamachat.model.LoadedModelInfo
import ru.npu.llamachat.model.ModelConfig
import ru.npu.llamachat.model.RamStats

/**
 * Корутинная обёртка над NpuLlamaBridge, SnapdragonNpuDetector и NpuMemoryBudgetEvaluator.
 * Гарантирует строгое выполнение исключительно на Qualcomm Hexagon NPU (HTP0)
 * без фоллбэка на CPU/GPU и без аварийного завершения (вылета) при выборе слишком больших моделей.
 */
data class NpuHardwareStatus(
    val ready: Boolean,
    val socReport: SocNpuReport,
    val hexagonArch: Int = 0,
    val htpDeviceCount: Int = 1,
    val hmxEnabled: Boolean = false,
    val activeDevices: String = "",
    val hwSummary: String = "",
    val errorMessage: String = ""
)

class ModelLoadRejectedException(
    message: String,
    val keptPreviousModel: Boolean
) : IllegalStateException(message)

class NpuChatEngine(context: Context) {

    private val appContext = context.applicationContext

    @Volatile
    private var lastHexagonArch: Int = 73

    @Volatile
    private var lastHtpDeviceCount: Int = 1

    suspend fun initializeNpu(config: ModelConfig): NpuHardwareStatus = withContext(Dispatchers.IO) {
        val socReport = SnapdragonNpuDetector.inspectCurrentDevice()
        if (socReport.blockingReason != null) {
            return@withContext NpuHardwareStatus(
                ready = false,
                socReport = socReport,
                errorMessage = socReport.blockingReason
            )
        }

        if (!NpuLlamaBridge.ensureLibrariesLoaded()) {
            return@withContext NpuHardwareStatus(
                ready = false,
                socReport = socReport,
                errorMessage = NpuLlamaBridge.libraryLoadError
                    ?: "Не удалось загрузить нативные библиотеки Hexagon NPU."
            )
        }

        val cleanCfg = config.sanitized()
        val nativeLibDir = appContext.applicationInfo.nativeLibraryDir
        val initRes = NpuLlamaBridge.nativeInitNpu(
            nativeLibDir = nativeLibDir,
            htpDevices = cleanCfg.htpDevices,
            powerMode = cleanCfg.powerMode,
            allowFp16Hmx = socReport.supportsFp16Hmx
        )

        val resolvedArch = if (initRes.hexagonArch > 0) initRes.hexagonArch else socReport.expectedHexagonArch
        if (resolvedArch > 0) lastHexagonArch = resolvedArch
        if (initRes.htpDeviceCount > 0) lastHtpDeviceCount = initRes.htpDeviceCount

        if (!initRes.ok) {
            return@withContext NpuHardwareStatus(
                ready = false,
                socReport = socReport,
                hexagonArch = resolvedArch,
                htpDeviceCount = initRes.htpDeviceCount,
                hmxEnabled = initRes.hmxEnabled,
                activeDevices = initRes.activeDevices,
                hwSummary = initRes.hwSummary,
                errorMessage = initRes.errorMessage
            )
        }

        NpuHardwareStatus(
            ready = true,
            socReport = socReport,
            hexagonArch = resolvedArch,
            htpDeviceCount = initRes.htpDeviceCount.coerceAtLeast(1),
            hmxEnabled = initRes.hmxEnabled,
            activeDevices = initRes.activeDevices,
            hwSummary = initRes.hwSummary
        )
    }

    suspend fun loadModelFromUri(
        uri: Uri,
        config: ModelConfig,
        hasLoadedModel: Boolean,
        onProgress: (Float) -> Unit
    ): Result<LoadedModelInfo> = withContext(Dispatchers.IO) {
        if (!NpuLlamaBridge.ensureLibrariesLoaded()) {
            return@withContext Result.failure(
                ModelLoadRejectedException(
                    NpuLlamaBridge.libraryLoadError ?: "Библиотеки NPU не загружены.",
                    keptPreviousModel = hasLoadedModel
                )
            )
        }

        val (displayName, queriedSize) = resolveFileMetadata(uri)
        if (!displayName.lowercase().endsWith(".gguf")) {
            return@withContext Result.failure(
                ModelLoadRejectedException(
                    "Выбран файл «$displayName». Пожалуйста, выберите файл модели в формате .gguf.",
                    keptPreviousModel = hasLoadedModel
                )
            )
        }

        // Снимок текущей памяти для двухэтапного контроля вместимости NPU и ОЗУ
        val (ramSnap, _) = readRamTelemetry()
        val reclaimableMib = ramSnap.npuHtpBufferMib + ramSnap.hostWeightMib

        // ЭТАП 1 (до открытия дескриптора/выделения буферов): быстрая проверка по размеру SAF-файла
        if (queriedSize > 0L) {
            val precheck = NpuMemoryBudgetEvaluator.checkBeforeNativeLoad(
                fileName = displayName,
                fileSizeBytes = queriedSize,
                hexagonArch = lastHexagonArch,
                activeHtpDeviceCount = lastHtpDeviceCount,
                systemAvailMib = ramSnap.systemAvailMib,
                systemTotalMib = ramSnap.systemTotalMib,
                reclaimableLoadedModelMib = reclaimableMib
            )
            if (!precheck.allowed) {
                return@withContext Result.failure(
                    ModelLoadRejectedException(
                        precheck.errorMessage,
                        keptPreviousModel = hasLoadedModel
                    )
                )
            }
        }

        val pfd = runCatching {
            appContext.contentResolver.openFileDescriptor(uri, "r")
        }.getOrNull()
            ?: return@withContext Result.failure(
                ModelLoadRejectedException(
                    "Не удалось открыть доступ к файлу «$displayName».",
                    keptPreviousModel = hasLoadedModel
                )
            )

        val statBytes = pfd.statSize.takeIf { it > 0L } ?: queriedSize
        if (statBytes > 0L && statBytes != queriedSize) {
            val fdPrecheck = NpuMemoryBudgetEvaluator.checkBeforeNativeLoad(
                fileName = displayName,
                fileSizeBytes = statBytes,
                hexagonArch = lastHexagonArch,
                activeHtpDeviceCount = lastHtpDeviceCount,
                systemAvailMib = ramSnap.systemAvailMib,
                systemTotalMib = ramSnap.systemTotalMib,
                reclaimableLoadedModelMib = reclaimableMib
            )
            if (!fdPrecheck.allowed) {
                runCatching { pfd.close() }
                return@withContext Result.failure(
                    ModelLoadRejectedException(
                        fdPrecheck.errorMessage,
                        keptPreviousModel = hasLoadedModel
                    )
                )
            }
        }

        // Передаём владение дескриптором в C++ (fdopen + fclose гарантируют освобождение ресурса;
        // внутри C++ сначала выполняется Этап 2 — zero-alloc разбор заголовков GGUF и расчёт тайлинга HTP)
        val detachedFd = pfd.detachFd()

        val cfg = config.sanitized()
        val nativeRes = NpuLlamaBridge.nativeLoadModelFromFd(
            detachedFd = detachedFd,
            displayName = displayName,
            nCtx = cfg.nCtx,
            maxNewTokens = cfg.maxNewTokens,
            nBatch = cfg.nBatch,
            nThreads = cfg.nThreads,
            nNpuLayers = cfg.nNpuLayers,
            temperature = cfg.temperature,
            topP = cfg.topP,
            topK = cfg.topK,
            minP = cfg.minP,
            repeatPenalty = cfg.repeatPenalty,
            repeatLastN = cfg.repeatLastN,
            seed = cfg.seed,
            flashAttnMode = cfg.flashAttnMode,
            kvQ8 = cfg.kvCacheQ8,
            enableThinking = cfg.enableThinking,
            powerMode = cfg.powerMode,
            htpDevices = cfg.htpDevices,
            systemPrompt = cfg.systemPrompt,
            progressListener = NpuLlamaBridge.ProgressListener { p -> onProgress(p) }
        )

        if (!nativeRes.ok) {
            return@withContext Result.failure(
                ModelLoadRejectedException(
                    nativeRes.errorMessage,
                    keptPreviousModel = nativeRes.keptPreviousModel
                )
            )
        }

        Result.success(
            LoadedModelInfo(
                fileName = displayName,
                modelDesc = nativeRes.modelDesc,
                quantName = nativeRes.quantName,
                fileSizeBytes = if (statBytes > 0L) statBytes else nativeRes.modelBytes,
                paramCount = nativeRes.modelParams,
                nCtxTrain = nativeRes.nCtxTrain,
                nCtxActive = nativeRes.nCtxActive,
                nLayersNpu = nativeRes.nLayersNpu,
                nLayersTotal = nativeRes.nLayersTotal,
                htpBufferMib = nativeRes.htpBufferMib,
                hostBufferMib = nativeRes.hostBufferMib,
                npuDeviceUsed = nativeRes.npuDeviceUsed,
                quantWarning = nativeRes.quantWarning
            )
        )
    }

    suspend fun applyConfig(config: ModelConfig): Result<Unit> = withContext(Dispatchers.IO) {
        if (!NpuLlamaBridge.ensureLibrariesLoaded()) {
            return@withContext Result.success(Unit)
        }
        val cfg = config.sanitized()
        val err = NpuLlamaBridge.nativeApplyConfig(
            nCtx = cfg.nCtx,
            maxNewTokens = cfg.maxNewTokens,
            nBatch = cfg.nBatch,
            nThreads = cfg.nThreads,
            nNpuLayers = cfg.nNpuLayers,
            temperature = cfg.temperature,
            topP = cfg.topP,
            topK = cfg.topK,
            minP = cfg.minP,
            repeatPenalty = cfg.repeatPenalty,
            repeatLastN = cfg.repeatLastN,
            seed = cfg.seed,
            flashAttnMode = cfg.flashAttnMode,
            kvQ8 = cfg.kvCacheQ8,
            enableThinking = cfg.enableThinking,
            powerMode = cfg.powerMode,
            htpDevices = cfg.htpDevices,
            systemPrompt = cfg.systemPrompt
        )
        if (err.isNotBlank()) {
            Result.failure(IllegalStateException(err))
        } else {
            Result.success(Unit)
        }
    }

    suspend fun generateReply(
        history: List<ChatMessage>,
        config: ModelConfig,
        onTokenDelta: (String) -> Boolean
    ): NativeGenerationStats = withContext(Dispatchers.IO) {
        val cfg = config.sanitized()
        val rolesList = ArrayList<String>(history.size + 1)
        val contentsList = ArrayList<String>(history.size + 1)

        if (cfg.systemPrompt.isNotBlank()) {
            rolesList.add("system")
            contentsList.add(cfg.systemPrompt.trim())
        }
        for (m in history) {
            if (m.content.isNotEmpty()) {
                rolesList.add(m.role.wireRole)
                contentsList.add(m.content)
            }
        }

        NpuLlamaBridge.nativeGenerate(
            roles = rolesList.toTypedArray(),
            contents = contentsList.toTypedArray(),
            nCtx = cfg.nCtx,
            maxNewTokens = cfg.maxNewTokens,
            nBatch = cfg.nBatch,
            nThreads = cfg.nThreads,
            nNpuLayers = cfg.nNpuLayers,
            temperature = cfg.temperature,
            topP = cfg.topP,
            topK = cfg.topK,
            minP = cfg.minP,
            repeatPenalty = cfg.repeatPenalty,
            repeatLastN = cfg.repeatLastN,
            seed = cfg.seed,
            flashAttnMode = cfg.flashAttnMode,
            kvQ8 = cfg.kvCacheQ8,
            enableThinking = cfg.enableThinking,
            powerMode = cfg.powerMode,
            htpDevices = cfg.htpDevices,
            systemPrompt = cfg.systemPrompt,
            tokenListener = NpuLlamaBridge.TokenStreamListener { delta -> onTokenDelta(delta) }
        )
    }

    fun cancelGeneration() {
        if (NpuLlamaBridge.ensureLibrariesLoaded()) {
            NpuLlamaBridge.nativeCancelGeneration()
        }
    }

    suspend fun clearChat() = withContext(Dispatchers.IO) {
        if (NpuLlamaBridge.ensureLibrariesLoaded()) {
            NpuLlamaBridge.nativeClearChat()
        }
    }

    suspend fun unloadModel() = withContext(Dispatchers.IO) {
        if (NpuLlamaBridge.ensureLibrariesLoaded()) {
            NpuLlamaBridge.nativeUnloadModel()
        }
    }

    suspend fun countTokens(text: String): Int = withContext(Dispatchers.Default) {
        if (text.isEmpty() || !NpuLlamaBridge.ensureLibrariesLoaded()) return@withContext 0
        NpuLlamaBridge.nativeCountTokens(text)
    }

    fun readRamTelemetry(): Pair<RamStats, Pair<Int, Int>> {
        val am = appContext.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        am?.getMemoryInfo(memInfo)

        if (!NpuLlamaBridge.ensureLibrariesLoaded()) {
            val sysTotalMib = memInfo.totalMem.toDouble() / (1024.0 * 1024.0)
            val sysAvailMib = memInfo.availMem.toDouble() / (1024.0 * 1024.0)
            val jvmUsedMib = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory())
                .toDouble() / (1024.0 * 1024.0)
            return RamStats(
                processRssMib = jvmUsedMib,
                processPeakRssMib = jvmUsedMib,
                systemAvailMib = sysAvailMib,
                systemTotalMib = sysTotalMib,
                isLowMemory = memInfo.lowMemory
            ) to (0 to 0)
        }

        val nativeMem = NpuLlamaBridge.nativeGetMemoryInfo()
        val sysTotal = if (nativeMem.systemTotalMib > 0.0) {
            nativeMem.systemTotalMib
        } else {
            memInfo.totalMem.toDouble() / (1024.0 * 1024.0)
        }
        val sysAvail = if (nativeMem.systemAvailMib > 0.0) {
            nativeMem.systemAvailMib
        } else {
            memInfo.availMem.toDouble() / (1024.0 * 1024.0)
        }

        val stats = RamStats(
            processRssMib = nativeMem.processRssMib,
            processPeakRssMib = nativeMem.processPeakRssMib,
            processSwapMib = nativeMem.processSwapMib,
            npuHtpBufferMib = nativeMem.npuHtpBufferMib,
            hostWeightMib = nativeMem.hostWeightMib,
            systemAvailMib = sysAvail,
            systemTotalMib = sysTotal,
            isLowMemory = memInfo.lowMemory
        )
        return stats to (nativeMem.contextUsedTokens to nativeMem.contextMaxTokens)
    }

    private fun resolveFileMetadata(uri: Uri): Pair<String, Long> {
        var name: String? = null
        var size = 0L
        runCatching {
            appContext.contentResolver.query(
                uri,
                arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE),
                null,
                null,
                null
            )?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val nameIdx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (nameIdx >= 0 && !cursor.isNull(nameIdx)) {
                        name = cursor.getString(nameIdx)
                    }
                    val sizeIdx = cursor.getColumnIndex(OpenableColumns.SIZE)
                    if (sizeIdx >= 0 && !cursor.isNull(sizeIdx)) {
                        size = cursor.getLong(sizeIdx)
                    }
                }
            }
        }
        val finalName = name?.takeIf { it.isNotBlank() }
            ?: uri.lastPathSegment?.substringAfterLast('/')
            ?: "model.gguf"
        return finalName to size
    }
}

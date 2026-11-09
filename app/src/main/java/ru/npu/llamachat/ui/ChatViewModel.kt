package ru.npu.llamachat.ui

import android.app.Application
import android.net.Uri
import android.os.SystemClock
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import ru.npu.llamachat.engine.ModelLoadRejectedException
import ru.npu.llamachat.engine.NpuChatEngine
import ru.npu.llamachat.engine.NpuHardwareStatus
import ru.npu.llamachat.engine.SnapdragonNpuDetector
import ru.npu.llamachat.model.ChatMessage
import ru.npu.llamachat.model.LoadedModelInfo
import ru.npu.llamachat.model.MessageRole
import ru.npu.llamachat.model.ModelConfig
import ru.npu.llamachat.model.RamStats
import ru.npu.llamachat.model.TokenStats
import java.util.concurrent.atomic.AtomicLong

data class UiAlertDialogEvent(
    val title: String,
    val message: String,
    val offerSettingsAction: Boolean = false
)

data class ChatUiState(
    val npuInitializing: Boolean = true,
    val npuStatus: NpuHardwareStatus = NpuHardwareStatus(
        ready = false,
        socReport = SnapdragonNpuDetector.inspectCurrentDevice()
    ),
    val config: ModelConfig = ModelConfig(),
    val isLoadingModel: Boolean = false,
    val isUnloadingModel: Boolean = false,
    val loadProgress: Float = 0f,
    val loadedModel: LoadedModelInfo? = null,
    val isGenerating: Boolean = false,
    /** Эфемерный список сообщений — хранится ТОЛЬКО в ОЗУ, никогда не сохраняется на диск. */
    val messages: List<ChatMessage> = emptyList(),
    val tokenStats: TokenStats = TokenStats(),
    val ramStats: RamStats = RamStats()
)

class ChatViewModel(application: Application) : AndroidViewModel(application) {

    private val engine = NpuChatEngine(application)
    private val nextMessageId = AtomicLong(1L)

    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    private val _events = MutableSharedFlow<String>(extraBufferCapacity = 8)
    val events: SharedFlow<String> = _events.asSharedFlow()

    private val _alertDialogs = MutableSharedFlow<UiAlertDialogEvent>(extraBufferCapacity = 4)
    val alertDialogs: SharedFlow<UiAlertDialogEvent> = _alertDialogs.asSharedFlow()

    private var generationJob: Job? = null
    private var modelLoadJob: Job? = null
    private var draftCountJob: Job? = null

    init {
        initHardwareBackend()
        startRamTelemetryLoop()
    }

    fun initHardwareBackend() {
        viewModelScope.launch {
            _uiState.update { it.copy(npuInitializing = true) }
            val status = engine.initializeNpu(_uiState.value.config)
            _uiState.update {
                it.copy(
                    npuInitializing = false,
                    npuStatus = status
                )
            }
            refreshRamOnce()
        }
    }

    private fun startRamTelemetryLoop() {
        viewModelScope.launch(Dispatchers.Default) {
            while (isActive) {
                refreshRamOnce()
                delay(1500L)
            }
        }
    }

    private fun refreshRamOnce() {
        val (ram, ctxPair) = engine.readRamTelemetry()
        val (ctxUsed, ctxMax) = ctxPair
        _uiState.update { cur ->
            val updatedTok = cur.tokenStats.copy(
                contextUsedTokens = if (cur.loadedModel != null) ctxUsed else 0,
                contextMaxTokens = if (ctxMax > 0) ctxMax else cur.config.nCtx
            )
            cur.copy(
                ramStats = ram,
                tokenStats = updatedTok
            )
        }
    }

    fun onDraftChanged(draft: String) {
        draftCountJob?.cancel()
        if (draft.isBlank() || _uiState.value.loadedModel == null) {
            _uiState.update { it.copy(tokenStats = it.tokenStats.copy(draftInputTokens = 0)) }
            return
        }
        draftCountJob = viewModelScope.launch {
            delay(120L)
            val n = engine.countTokens(draft)
            _uiState.update { it.copy(tokenStats = it.tokenStats.copy(draftInputTokens = n)) }
        }
    }

    fun loadModel(uri: Uri) {
        val state = _uiState.value
        if (!state.npuStatus.ready) {
            _alertDialogs.tryEmit(
                UiAlertDialogEvent(
                    title = "NPU недоступен",
                    message = state.npuStatus.errorMessage.ifBlank {
                        "Загрузка заблокирована: Qualcomm Hexagon NPU недоступен."
                    }
                )
            )
            return
        }
        if (state.isLoadingModel || state.isUnloadingModel) return

        val hadModelBefore = state.loadedModel != null
        generationJob?.cancel()
        engine.cancelGeneration()

        modelLoadJob = viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isLoadingModel = true,
                    loadProgress = 0f,
                    isGenerating = false
                )
            }

            val result = engine.loadModelFromUri(
                uri = uri,
                config = _uiState.value.config,
                hasLoadedModel = hadModelBefore,
                onProgress = { prog ->
                    _uiState.update { s -> s.copy(loadProgress = prog.coerceIn(0f, 1f)) }
                }
            )

            result.fold(
                onSuccess = { info ->
                    // При смене модели очищаем эфемерный чат и счётчики предыдущей сессии
                    _uiState.update { s ->
                        s.copy(
                            isLoadingModel = false,
                            loadProgress = 1f,
                            loadedModel = info,
                            messages = emptyList(),
                            tokenStats = TokenStats(
                                contextUsedTokens = 0,
                                contextMaxTokens = info.nCtxActive
                            )
                        )
                    }
                    refreshRamOnce()
                    val baseToast = "Модель «${info.fileName}» загружена на ${info.npuDeviceUsed} " +
                        "(${info.nLayersNpu}/${info.nLayersTotal} слоёв на NPU)"
                    _events.tryEmit(baseToast)
                },
                onFailure = { err ->
                    val keptPrev = (err as? ModelLoadRejectedException)?.keptPreviousModel ?: false
                    _uiState.update { s ->
                        s.copy(
                            isLoadingModel = false,
                            loadProgress = 0f,
                            loadedModel = if (keptPrev) s.loadedModel else null
                        )
                    }
                    refreshRamOnce()
                    val errMsg = err.message ?: "Не удалось загрузить файл .gguf в память Hexagon NPU."
                    val suggestSettings = errMsg.contains("n_ctx", ignoreCase = true) ||
                        errMsg.contains("контекст", ignoreCase = true)
                    _alertDialogs.tryEmit(
                        UiAlertDialogEvent(
                            title = "Модель не загружена в Hexagon NPU",
                            message = errMsg,
                            offerSettingsAction = suggestSettings
                        )
                    )
                }
            )
        }
    }

    /** Освобождает веса и KV-кэш, не завершая приложение. */
    fun unloadModel() {
        val state = _uiState.value
        if (state.loadedModel == null || state.isLoadingModel || state.isUnloadingModel) return
        _uiState.update { it.copy(isUnloadingModel = true) }
        engine.cancelGeneration()
        generationJob?.cancel()
        draftCountJob?.cancel()
        viewModelScope.launch {
            generationJob?.join()
            engine.unloadModel()
            _uiState.update {
                it.copy(
                    isUnloadingModel = false,
                    isGenerating = false,
                    loadedModel = null,
                    messages = emptyList(),
                    tokenStats = TokenStats(contextMaxTokens = it.config.nCtx)
                )
            }
            refreshRamOnce()
            _events.tryEmit("Модель выгружена из NPU")
        }
    }

    fun cancelActiveTask() {
        engine.cancelGeneration()
    }

    fun sendMessage(rawUserText: String) {
        val text = rawUserText.trim()
        if (text.isEmpty()) return

        val cur = _uiState.value
        if (!cur.npuStatus.ready) {
            _events.tryEmit("Отправка заблокирована: нейропроцессор Qualcomm Hexagon NPU не активен.")
            return
        }
        if (cur.loadedModel == null) {
            _events.tryEmit("Сначала выберите файл модели .gguf.")
            return
        }
        if (cur.isGenerating || cur.isLoadingModel || cur.isUnloadingModel) {
            return
        }

        generationJob = viewModelScope.launch {
            val userTokens = engine.countTokens(text)
            val userMsg = ChatMessage(
                id = nextMessageId.getAndIncrement(),
                role = MessageRole.USER,
                content = text,
                tokenCount = userTokens
            )
            val assistantId = nextMessageId.getAndIncrement()
            val assistantPlaceholder = ChatMessage(
                id = assistantId,
                role = MessageRole.ASSISTANT,
                content = "",
                isStreaming = true
            )

            val historyForPrompt = cur.messages + userMsg

            _uiState.update { s ->
                s.copy(
                    isGenerating = true,
                    messages = historyForPrompt + assistantPlaceholder,
                    tokenStats = s.tokenStats.copy(draftInputTokens = 0)
                )
            }

            val fullReply = StringBuilder()
            var streamedTokensApprox = 0
            var lastUiPushMs = 0L

            val stats = engine.generateReply(
                history = historyForPrompt,
                config = _uiState.value.config,
                onTokenDelta = { delta ->
                    fullReply.append(delta)
                    streamedTokensApprox += 1
                    val now = SystemClock.uptimeMillis()
                    if (now - lastUiPushMs >= 32L) {
                        lastUiPushMs = now
                        val snapshot = fullReply.toString()
                        val approx = streamedTokensApprox
                        _uiState.update { s ->
                            s.copy(
                                messages = s.messages.map { m ->
                                    if (m.id == assistantId) {
                                        m.copy(content = snapshot, tokenCount = approx, isStreaming = true)
                                    } else {
                                        m
                                    }
                                }
                            )
                        }
                    }
                    true
                }
            )

            val finalContent = fullReply.toString()
            if (!stats.ok) {
                _uiState.update { s ->
                    s.copy(
                        isGenerating = false,
                        messages = s.messages.filterNot { it.id == assistantId && finalContent.isEmpty() }.map { m ->
                            if (m.id == assistantId) m.copy(content = finalContent, isStreaming = false) else m
                        }
                    )
                }
                val errText = stats.errorMessage.ifBlank { "Ошибка генерации на Hexagon NPU." }
                _alertDialogs.tryEmit(
                    UiAlertDialogEvent(
                        title = "Ошибка генерации на Hexagon NPU",
                        message = errText
                    )
                )
                refreshRamOnce()
                return@launch
            }

            _uiState.update { s ->
                val updatedMessages = s.messages.map { m ->
                    if (m.id == assistantId) {
                        m.copy(
                            content = finalContent.ifEmpty { if (stats.stoppedByUser) "⏹ Остановлено" else "" },
                            tokenCount = stats.generatedTokens,
                            tokPerSec = stats.decodeTokPerSec,
                            isStreaming = false,
                            wasStopped = stats.stoppedByUser
                        )
                    } else {
                        m
                    }
                }
                s.copy(
                    isGenerating = false,
                    messages = updatedMessages,
                    tokenStats = s.tokenStats.copy(
                        contextUsedTokens = stats.contextUsedTokens,
                        contextMaxTokens = stats.contextMaxTokens.takeIf { it > 0 } ?: s.config.nCtx,
                        lastPromptTokens = stats.promptTokens,
                        lastCachedTokens = stats.cachedTokens,
                        lastGeneratedTokens = stats.generatedTokens,
                        sessionGeneratedTokens = s.tokenStats.sessionGeneratedTokens + stats.generatedTokens,
                        prefillTokPerSec = stats.prefillTokPerSec,
                        decodeTokPerSec = stats.decodeTokPerSec
                    )
                )
            }
            refreshRamOnce()
        }
    }

    /**
     * Полная очистка эфемерного чата и сброс KV-кэша на Hexagon NPU.
     */
    fun clearChat() {
        engine.cancelGeneration()
        generationJob?.cancel()
        viewModelScope.launch {
            engine.clearChat()
            _uiState.update { s ->
                s.copy(
                    isGenerating = false,
                    messages = emptyList(),
                    tokenStats = TokenStats(
                        contextUsedTokens = 0,
                        contextMaxTokens = s.loadedModel?.nCtxActive ?: s.config.nCtx,
                        draftInputTokens = s.tokenStats.draftInputTokens
                    )
                )
            }
            refreshRamOnce()
            _events.tryEmit("Чат и KV-кэш NPU очищены")
        }
    }

    fun updateModelConfig(newConfig: ModelConfig) {
        val clean = newConfig.sanitized()
        val prev = _uiState.value.config
        val contextRecreated = _uiState.value.loadedModel != null && (
            clean.nCtx != prev.nCtx ||
                clean.nBatch != prev.nBatch ||
                clean.flashAttnMode != prev.flashAttnMode ||
                clean.kvCacheQ8 != prev.kvCacheQ8
            )
        val thinkReset = _uiState.value.loadedModel != null &&
            clean.enableThinking != prev.enableThinking

        engine.cancelGeneration()
        generationJob?.cancel()

        viewModelScope.launch {
            val res = engine.applyConfig(clean)
            res.fold(
                onSuccess = {
                    _uiState.update { s ->
                        val updLoaded = s.loadedModel?.copy(nCtxActive = clean.nCtx)
                        s.copy(
                            isGenerating = false,
                            config = clean,
                            loadedModel = updLoaded,
                            tokenStats = s.tokenStats.copy(
                                contextMaxTokens = clean.nCtx,
                                contextUsedTokens = if (contextRecreated || thinkReset) 0 else s.tokenStats.contextUsedTokens
                            )
                        )
                    }
                    refreshRamOnce()
                    _events.tryEmit(
                        if (contextRecreated) {
                            "Параметры применены (контекст NPU пересоздан: ${clean.nCtx} токенов)"
                        } else {
                            "Параметры модели сохранены для текущей сессии"
                        }
                    )
                },
                onFailure = { err ->
                    _alertDialogs.tryEmit(
                        UiAlertDialogEvent(
                            title = "Не удалось применить параметры NPU",
                            message = err.message ?: "Ошибка применения конфигурации модели.",
                            offerSettingsAction = true
                        )
                    )
                }
            )
        }
    }

    override fun onCleared() {
        engine.cancelGeneration()
        super.onCleared()
    }
}

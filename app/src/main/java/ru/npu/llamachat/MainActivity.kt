package ru.npu.llamachat

import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.View
import android.view.inputmethod.EditorInfo
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import com.google.android.material.snackbar.Snackbar
import kotlinx.coroutines.launch
import ru.npu.llamachat.databinding.ActivityMainBinding
import ru.npu.llamachat.ui.ChatAdapter
import ru.npu.llamachat.ui.ChatUiState
import ru.npu.llamachat.ui.ChatViewModel
import ru.npu.llamachat.ui.SettingsBottomSheet
import java.util.Locale
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val viewModel: ChatViewModel by viewModels()
    private val chatAdapter = ChatAdapter()

    /**
     * Выбор .gguf файла через системный Storage Access Framework (DocumentsUI) без запроса лишних разрешений.
     */
    private val pickGgufLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            viewModel.loadModel(uri)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        applyWindowInsets()
        setupToolbar()
        setupRecyclerView()
        setupInputControls()
        observeViewModel()
    }

    private fun applyWindowInsets() {
        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { v, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.ime()
            )
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
    }

    private fun setupToolbar() {
        binding.topAppBar.subtitle = "v${BuildConfig.VERSION_NAME} • Только Hexagon NPU"
        binding.topAppBar.setOnMenuItemClickListener { item ->
            when (item.itemId) {
                R.id.action_pick_model -> {
                    launchGgufPicker()
                    true
                }
                R.id.action_clear_chat -> {
                    viewModel.clearChat()
                    true
                }
                R.id.action_model_settings -> {
                    openSettingsSheet()
                    true
                }
                else -> false
            }
        }
    }

    private fun setupRecyclerView() {
        val lm = LinearLayoutManager(this).apply {
            stackFromEnd = true
        }
        binding.rvMessages.layoutManager = lm
        binding.rvMessages.itemAnimator = null
        binding.rvMessages.adapter = chatAdapter
    }

    private fun setupInputControls() {
        binding.btnSelectGgufHeader.setOnClickListener { launchGgufPicker() }
        binding.btnPickModelEmpty.setOnClickListener { launchGgufPicker() }
        binding.btnOpenSettingsEmpty.setOnClickListener { openSettingsSheet() }
        binding.btnRetryNpuInit.setOnClickListener { viewModel.initHardwareBackend() }
        binding.btnClearChatQuick.setOnClickListener { viewModel.clearChat() }

        binding.etPromptInput.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) = Unit
            override fun afterTextChanged(s: Editable?) {
                viewModel.onDraftChanged(s?.toString().orEmpty())
            }
        })

        binding.etPromptInput.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_SEND) {
                triggerSendIfPossible()
                true
            } else {
                false
            }
        }

        binding.btnSendOrStop.setOnClickListener {
            val state = viewModel.uiState.value
            if (state.isGenerating || state.isLoadingModel) {
                viewModel.cancelActiveTask()
            } else {
                triggerSendIfPossible()
            }
        }
    }

    private fun triggerSendIfPossible() {
        val text = binding.etPromptInput.text?.toString().orEmpty().trim()
        if (text.isEmpty()) return
        binding.etPromptInput.setText("")
        viewModel.sendMessage(text)
    }

    private fun launchGgufPicker() {
        val state = viewModel.uiState.value
        if (!state.npuStatus.ready && !state.npuInitializing) {
            Snackbar.make(
                binding.root,
                "Загрузка GGUF заблокирована: требуется аппаратный Qualcomm Hexagon NPU",
                Snackbar.LENGTH_LONG
            ).show()
            return
        }
        pickGgufLauncher.launch(arrayOf("*/*"))
    }

    private fun openSettingsSheet() {
        if (supportFragmentManager.findFragmentByTag(SettingsBottomSheet.TAG) == null) {
            SettingsBottomSheet().show(supportFragmentManager, SettingsBottomSheet.TAG)
        }
    }

    private fun observeViewModel() {
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch {
                    viewModel.uiState.collect { state ->
                        renderUiState(state)
                    }
                }
                launch {
                    viewModel.events.collect { message ->
                        Snackbar.make(binding.root, message, Snackbar.LENGTH_SHORT).show()
                    }
                }
            }
        }
    }

    private fun renderUiState(state: ChatUiState) {
        // 1. NPU Hardware Badge & Strict CPU/GPU Block Banner
        val npu = state.npuStatus
        when {
            state.npuInitializing -> {
                binding.tvNpuBadge.text = "Проверка Qualcomm Hexagon NPU…"
                binding.cardNpuBlockedBanner.visibility = View.GONE
            }
            npu.ready -> {
                val archStr = if (npu.hexagonArch > 0) "v${npu.hexagonArch}" else "HTP"
                binding.tvNpuBadge.text =
                    "NPU АКТИВЕН • ${npu.socReport.marketingName} • ${npu.activeDevices} ($archStr)"
                binding.cardNpuBlockedBanner.visibility = View.GONE
            }
            else -> {
                binding.tvNpuBadge.text = "NPU НЕДОСТУПЕН • CPU/GPU ЗАБЛОКИРОВАНЫ"
                binding.cardNpuBlockedBanner.visibility = View.VISIBLE
                binding.tvNpuBlockedMessage.text = npu.errorMessage
            }
        }

        // 2. Loaded Model Status & Progress
        val model = state.loadedModel
        if (state.isLoadingModel) {
            val pct = (state.loadProgress * 100f).roundToInt()
            binding.progressModelLoad.visibility = View.VISIBLE
            binding.progressModelLoad.setProgressCompat(pct, true)
            binding.tvLoadProgressLabel.visibility = View.VISIBLE
            binding.tvLoadProgressLabel.text = "Загрузка весов в Hexagon NPU RPCMEM: $pct%"
            binding.btnSelectGgufHeader.isEnabled = false
        } else {
            binding.progressModelLoad.visibility = View.GONE
            binding.tvLoadProgressLabel.visibility = View.GONE
            binding.btnSelectGgufHeader.isEnabled = npu.ready
        }

        if (model != null) {
            val sizeMib = model.fileSizeBytes.toDouble() / (1024.0 * 1024.0)
            binding.tvModelBadge.text = String.format(
                Locale.US,
                "%s [%s • %.0f МБ] • %d/%d слоёв на %s",
                model.fileName,
                model.quantName,
                sizeMib,
                model.nLayersNpu,
                model.nLayersTotal,
                model.npuDeviceUsed
            )
            binding.btnSelectGgufHeader.text = getString(R.string.action_change_gguf)
        } else {
            binding.tvModelBadge.text = getString(R.string.label_no_model_loaded)
            binding.btnSelectGgufHeader.text = getString(R.string.action_select_gguf)
        }

        // 3. Token Telemetry Strip
        val tok = state.tokenStats
        val ctxMax = tok.contextMaxTokens.coerceAtLeast(1)
        val ctxPct = ((tok.contextUsedTokens.toFloat() / ctxMax.toFloat()) * 100f)
            .roundToInt()
            .coerceIn(0, 100)
        binding.progressContextUsage.setProgressCompat(ctxPct, true)

        val speedSuffix = if (tok.decodeTokPerSec > 0.05) {
            String.format(Locale.US, " • %.1f ток/с", tok.decodeTokPerSec)
        } else {
            ""
        }
        binding.tvTokenStatsLine.text =
            "Токены контекста: ${tok.contextUsedTokens} / $ctxMax ($ctxPct%) • " +
                "Промпт: ${tok.lastPromptTokens} (кэш ${tok.lastCachedTokens}) • " +
                "Ответ: ${tok.lastGeneratedTokens} (всего ${tok.sessionGeneratedTokens})$speedSuffix"

        // 4. Live RAM Consumption Telemetry Strip
        val ram = state.ramStats
        val sysAvailGib = ram.systemAvailMib / 1024.0
        val sysTotalGib = ram.systemTotalMib / 1024.0
        val htpPart = if (ram.npuHtpBufferMib > 0.1) {
            String.format(Locale.US, " • HTP0: %.0f МБ", ram.npuHtpBufferMib)
        } else {
            ""
        }
        binding.tvRamStatsLine.text = String.format(
            Locale.US,
            "ОЗУ процесса: %.0f МБ (пик %.0f МБ)%s • Свободно ОЗУ: %.1f / %.1f ГБ",
            ram.processRssMib,
            ram.processPeakRssMib,
            htpPart,
            sysAvailGib,
            sysTotalGib
        )

        // 5. Draft Token Counter & Clear Button
        binding.tvDraftTokens.text = if (tok.draftInputTokens > 0) {
            "Ввод: ${tok.draftInputTokens} ток."
        } else {
            "История чата не сохраняется"
        }
        binding.btnClearChatQuick.visibility =
            if (state.messages.isNotEmpty() || tok.contextUsedTokens > 0) View.VISIBLE else View.GONE

        // 6. Messages vs Empty State
        val prevCount = chatAdapter.itemCount
        chatAdapter.submitList(state.messages) {
            if (state.messages.isNotEmpty() && (state.messages.size != prevCount || state.isGenerating)) {
                binding.rvMessages.scrollToPosition(state.messages.size - 1)
            }
        }

        if (state.messages.isEmpty()) {
            binding.layoutEmptyState.visibility = View.VISIBLE
            binding.rvMessages.visibility = View.GONE
            binding.tvEmptySubtitle.text = when {
                !npu.ready && !state.npuInitializing ->
                    "Устройство не прошло проверку Qualcomm Hexagon NPU. Выполнение на CPU/GPU отключено."
                model == null ->
                    "Выберите локальный файл модели .gguf (рекомендуются Q4_0 / Q8_0 / Q4_K_M для Hexagon HTP v73 на Snapdragon 8s Gen 3). История диалога хранится только в ОЗУ и удаляется при закрытии."
                else ->
                    "Модель «${model.fileName}» готова к работе на ${model.npuDeviceUsed} (макс. контекст ${model.nCtxActive} токенов). Напишите сообщение ниже."
            }
            binding.btnPickModelEmpty.isEnabled = npu.ready && !state.isLoadingModel
        } else {
            binding.layoutEmptyState.visibility = View.GONE
            binding.rvMessages.visibility = View.VISIBLE
        }

        // 7. Input Field & Send / Stop Button
        val canChat = npu.ready && model != null && !state.isLoadingModel
        binding.etPromptInput.isEnabled = canChat && !state.isGenerating
        if (state.isGenerating || state.isLoadingModel) {
            binding.btnSendOrStop.isEnabled = true
            binding.btnSendOrStop.text = getString(R.string.action_stop)
        } else {
            binding.btnSendOrStop.isEnabled = canChat
            binding.btnSendOrStop.text = getString(R.string.action_send)
        }
    }

    override fun onDestroy() {
        if (isFinishing) {
            // При закрытии приложения гарантированно выгружаем буферы Hexagon NPU
            if (ru.npu.llamachat.engine.NpuLlamaBridge.ensureLibrariesLoaded()) {
                ru.npu.llamachat.engine.NpuLlamaBridge.nativeCancelGeneration()
                ru.npu.llamachat.engine.NpuLlamaBridge.nativeUnloadModel()
            }
        }
        super.onDestroy()
    }
}

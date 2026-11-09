package ru.npu.llamachat

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.view.inputmethod.EditorInfo
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.snackbar.Snackbar
import com.google.android.material.tabs.TabLayout
import ru.npu.llamachat.api.ApiService
import ru.npu.llamachat.image.ImageModelCatalog
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import ru.npu.llamachat.databinding.ActivityMainBinding
import ru.npu.llamachat.ui.ChatAdapter
import ru.npu.llamachat.ui.ChatUiState
import ru.npu.llamachat.ui.ChatViewModel
import ru.npu.llamachat.ui.SettingsBottomSheet
import ru.npu.llamachat.ui.UiAlertDialogEvent
import java.util.Locale
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val viewModel: ChatViewModel get() = (application as NpuApplication).session
    private val imageViewModel get() = (application as NpuApplication).images
    private val pickImageFolder = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        if (uri != null) imageViewModel.importModel(uri)
    }
    private val saveImage = registerForActivityResult(ActivityResultContracts.CreateDocument("image/png")) { uri ->
        val bitmap = imageViewModel.state.value.bitmap
        if (uri != null && bitmap != null) lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching { contentResolver.openOutputStream(uri)?.use { bitmap.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it) } == true }
            }
            Snackbar.make(binding.root, if (result.getOrDefault(false)) "PNG сохранён" else "Не удалось сохранить PNG", Snackbar.LENGTH_LONG).show()
        }
    }
    private val chatAdapter = ChatAdapter()
    private var activeAlertDialog: AlertDialog? = null

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

    private val notificationPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startApi() else Snackbar.make(binding.root, "Для управления API в фоне разрешите уведомления", Snackbar.LENGTH_LONG).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        applyWindowInsets()
        setupToolbar()
        setupTabsAndApi()
        setupRecyclerView()
        setupInputControls()
        observeViewModel()
        if (intent?.action == ApiService.LOAD) launchGgufPicker()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        if (intent.action == ApiService.LOAD) launchGgufPicker()
    }

    private fun setupTabsAndApi() {
        binding.tabs.addTab(binding.tabs.newTab().setText("Чат"))
        binding.tabs.addTab(binding.tabs.newTab().setText("API"))
        binding.tabs.addTab(binding.tabs.newTab().setText("Изображения"))
        binding.tabs.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab) {
                binding.chatPanel.visibility = if (tab.position == 0) View.VISIBLE else View.GONE
                binding.apiPanel.visibility = if (tab.position == 1) View.VISIBLE else View.GONE
                binding.imagePanel.visibility = if (tab.position == 2) View.VISIBLE else View.GONE
            }
            override fun onTabUnselected(tab: TabLayout.Tab) = Unit
            override fun onTabReselected(tab: TabLayout.Tab) = Unit
        })
        binding.spinnerImageModel.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item,
            ImageModelCatalog.models.map { it.title })
        binding.spinnerImageModel.setSelection(imageViewModel.state.value.selectedModel)
        binding.spinnerImageModel.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (position != imageViewModel.state.value.selectedModel) binding.etImageSteps.setText(if (position == 0) "4" else "30")
                imageViewModel.selectModel(position)
            }
            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }
        binding.btnImageDownload.setOnClickListener {
            val model = ImageModelCatalog.models[imageViewModel.state.value.selectedModel]
            val mib = model.totalBytes / (1024 * 1024)
            MaterialAlertDialogBuilder(this).setTitle("Скачать ${model.title}?")
                .setMessage("Будет скачано около $mib МиБ во внутреннюю память приложения. Нужны интернет и свободное место; при замене старой копии нужно место для обеих. Условия CreativeML OpenRAIL-M распространяются на модель." +
                    if (model == ImageModelCatalog.anything) " Внимание: NPU-совместимость Anything V5 не подтверждена." else "")
                .setNegativeButton("Отмена", null).setPositiveButton("Скачать") { _, _ -> imageViewModel.downloadModel() }.show()
        }
        binding.btnImageImport.setOnClickListener { pickImageFolder.launch(null) }
        binding.btnImageGenerate.setOnClickListener {
            val steps = binding.etImageSteps.text?.toString()?.toIntOrNull()
            val seedText = binding.etImageSeed.text?.toString().orEmpty().trim()
            val seed = if (seedText.isEmpty()) java.security.SecureRandom().nextLong() else seedText.toLongOrNull()
            val guidance = if (imageViewModel.state.value.selectedModel == 0) 1f else binding.etImageGuidance.text?.toString()?.toFloatOrNull()
            if (steps == null || seed == null || guidance == null) {
                Snackbar.make(binding.root, "Проверьте шаги, CFG и seed", Snackbar.LENGTH_LONG).show()
            } else imageViewModel.generate(binding.etImagePrompt.text?.toString().orEmpty(),
                binding.etNegativePrompt.text?.toString().orEmpty(), steps, guidance, seed)
        }
        binding.btnImageCancel.setOnClickListener { imageViewModel.cancel() }
        binding.btnImageSave.setOnClickListener { saveImage.launch("image.png") }
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                imageViewModel.state.collect { state ->
                    binding.tvImageStatus.text = state.status
                    binding.spinnerImageModel.isEnabled = !state.busy
                    binding.btnImageDownload.isEnabled = !state.busy
                    binding.btnImageImport.isEnabled = !state.busy && state.selectedModel == 0
                    binding.negativePromptLayout.visibility = if (state.selectedModel == 1) View.VISIBLE else View.GONE
                    binding.guidanceLayout.visibility = if (state.selectedModel == 1) View.VISIBLE else View.GONE
                    binding.btnImageGenerate.isEnabled = !state.busy && state.installed && state.canGenerate
                    binding.btnImageCancel.isEnabled = state.busy
                    binding.btnImageSave.isEnabled = !state.busy && state.bitmap != null
                    binding.ivGeneratedImage.setImageBitmap(state.bitmap)
                }
            }
        }
        binding.btnApiStart.setOnClickListener {
            if (Build.VERSION.SDK_INT >= 33 && ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
            } else startApi()
        }
        binding.btnApiStop.setOnClickListener { stopService(Intent(this, ApiService::class.java)) }
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch { ApiService.running.collect { running ->
                    binding.tvApiStatus.text = if (running) "API запущен: http://127.0.0.1:8765" else "API остановлен"
                    binding.btnApiStart.isEnabled = !running
                    binding.btnApiStop.isEnabled = running
                } }
                launch { ApiService.apiToken.collect { token -> binding.tvApiToken.text = if (token.isNotEmpty()) "Bearer token: $token" else "" } }
                launch { ApiService.error.collect { if (it.isNotBlank()) Snackbar.make(binding.root, it, Snackbar.LENGTH_LONG).show() } }
            }
        }
    }

    private fun startApi() {
        try {
            ContextCompat.startForegroundService(this, Intent(this, ApiService::class.java))
        } catch (e: Exception) {
            Snackbar.make(binding.root, "Не удалось запустить сервис: ${e.message}", Snackbar.LENGTH_LONG).show()
        }
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
                R.id.action_unload_model -> {
                    viewModel.unloadModel()
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
        if (state.isLoadingModel || state.isUnloadingModel) return
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

    private fun showDiagnosticDialog(event: UiAlertDialogEvent) {
        if (isFinishing || isDestroyed) return
        activeAlertDialog?.dismiss()
        val builder = MaterialAlertDialogBuilder(this)
            .setTitle(event.title)
            .setMessage(event.message)
            .setPositiveButton("Понятно", null)
        if (event.offerSettingsAction) {
            builder.setNeutralButton("Параметры NPU") { _, _ ->
                openSettingsSheet()
            }
        }
        activeAlertDialog = builder.show()
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
                launch {
                    viewModel.alertDialogs.collect { dialogEvent ->
                        showDiagnosticDialog(dialogEvent)
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
                val engineMode = if (npu.hmxEnabled) "HMX+HVX" else "HVX"
                binding.tvNpuBadge.text =
                    "NPU АКТИВЕН • ${npu.socReport.marketingName} • ${npu.activeDevices} ($archStr $engineMode)"
                binding.cardNpuBlockedBanner.visibility = View.GONE
            }
            else -> {
                binding.tvNpuBadge.text = "NPU НЕДОСТУПЕН • CPU/GPU ЗАБЛОКИРОВАНЫ"
                binding.cardNpuBlockedBanner.visibility = View.VISIBLE
                binding.tvNpuBlockedMessage.text = npu.errorMessage
            }
        }

        binding.topAppBar.menu.findItem(R.id.action_unload_model).isEnabled =
            state.loadedModel != null && !state.isLoadingModel && !state.isUnloadingModel && !state.isApiGenerating

        // 2. Loaded Model Status & Progress
        val model = state.loadedModel
        if (state.isLoadingModel) {
            val pct = (state.loadProgress * 100f).roundToInt()
            binding.progressModelLoad.visibility = View.VISIBLE
            binding.progressModelLoad.setProgressCompat(pct, true)
            binding.tvLoadProgressLabel.visibility = View.VISIBLE
            binding.tvLoadProgressLabel.text = "Проверка и загрузка весов в Hexagon NPU RPCMEM: $pct%"
            binding.btnSelectGgufHeader.isEnabled = false
        } else {
            binding.progressModelLoad.visibility = View.GONE
            binding.tvLoadProgressLabel.visibility = View.GONE
            binding.btnSelectGgufHeader.isEnabled = npu.ready && !state.isUnloadingModel
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
                    "Выберите локальный файл модели .gguf размера 0.5B–3B (до ~2.4 ГБ).\nОптимальное квантование для Hexagon HTP NPU: Q4_0 (int4) или Q8_0 (int8). История хранится только в ОЗУ."
                model.quantWarning.isNotBlank() ->
                    "Модель «${model.fileName}» готова на ${model.npuDeviceUsed} (контекст ${model.nCtxActive} ток.).\n💡 ${model.quantWarning}"
                else ->
                    "Модель «${model.fileName}» готова к работе на ${model.npuDeviceUsed} (макс. контекст ${model.nCtxActive} токенов). Напишите сообщение ниже."
            }
            binding.btnPickModelEmpty.isEnabled = npu.ready && !state.isLoadingModel && !state.isUnloadingModel
        } else {
            binding.layoutEmptyState.visibility = View.GONE
            binding.rvMessages.visibility = View.VISIBLE
        }

        // 7. Input Field & Send / Stop Button
        val canChat = npu.ready && model != null && !state.isLoadingModel && !state.isUnloadingModel && !state.isApiGenerating
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
        activeAlertDialog?.dismiss()
        activeAlertDialog = null
        if (isFinishing && !ru.npu.llamachat.api.ApiService.running.value) {
            viewModel.unloadModel()
        }
        super.onDestroy()
    }
}

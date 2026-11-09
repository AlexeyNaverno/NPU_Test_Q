package ru.npu.llamachat.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.activityViewModels
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import ru.npu.llamachat.R
import ru.npu.llamachat.databinding.SheetModelSettingsBinding
import ru.npu.llamachat.model.ModelConfig
import java.util.Locale
import kotlin.math.roundToInt

/**
 * Диалог настройки максимального контекста, параметров сэмплинга и аппаратного профиля Hexagon NPU.
 */
class SettingsBottomSheet : BottomSheetDialogFragment() {

    private var _binding: SheetModelSettingsBinding? = null
    private val binding get() = _binding!!

    private val viewModel: ChatViewModel by activityViewModels()

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = SheetModelSettingsBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        (dialog as? BottomSheetDialog)?.behavior?.apply {
            state = BottomSheetBehavior.STATE_EXPANDED
            skipCollapsed = true
        }

        populateFromConfig(viewModel.uiState.value.config)
        setupInteractiveListeners()
    }

    private fun populateFromConfig(cfg: ModelConfig) {
        binding.etMaxContext.setText(cfg.nCtx.toString())
        syncContextChips(cfg.nCtx)

        binding.etMaxNewTokens.setText(cfg.maxNewTokens.toString())
        syncBatchChips(cfg.nBatch)

        val tempClamped = ((cfg.temperature * 20f).roundToInt() / 20f).coerceIn(0f, 2f)
        binding.sliderTemperature.value = tempClamped
        binding.tvTemperatureVal.text = String.format(Locale.US, "%.2f", tempClamped)

        val topPClamped = ((cfg.topP * 20f).roundToInt() / 20f).coerceIn(0.10f, 1.0f)
        binding.sliderTopP.value = topPClamped
        binding.tvTopPVal.text = String.format(Locale.US, "%.2f", topPClamped)

        binding.etTopK.setText(cfg.topK.toString())

        val minPClamped = ((cfg.minP * 100f).roundToInt() / 100f).coerceIn(0.0f, 0.50f)
        binding.sliderMinP.value = minPClamped
        binding.tvMinPVal.text = String.format(Locale.US, "%.2f", minPClamped)

        val repClamped = ((cfg.repeatPenalty * 20f).roundToInt() / 20f).coerceIn(1.0f, 2.0f)
        binding.sliderRepeatPenalty.value = repClamped
        binding.tvRepeatPenaltyVal.text = String.format(Locale.US, "%.2f", repClamped)

        binding.etSeed.setText(cfg.seed.toString())

        val threadsClamped = cfg.nThreads.coerceIn(1, 8).toFloat()
        binding.sliderThreads.value = threadsClamped
        binding.tvThreadsVal.text = cfg.nThreads.coerceIn(1, 8).toString()

        when (cfg.powerMode) {
            ModelConfig.POWER_BURST -> binding.rgPowerMode.check(R.id.rbPowerBurst)
            ModelConfig.POWER_BALANCED -> binding.rgPowerMode.check(R.id.rbPowerBalanced)
            ModelConfig.POWER_SAVER -> binding.rgPowerMode.check(R.id.rbPowerSaver)
            else -> binding.rgPowerMode.check(R.id.rbPowerSustained)
        }

        binding.switchFlashAttn.isChecked = cfg.flashAttnMode > 0
        binding.switchKvQ8.isChecked = cfg.kvCacheQ8
        binding.switchThinkingMode.isChecked = cfg.enableThinking
        binding.etSystemPrompt.setText(cfg.systemPrompt)
    }

    private fun syncContextChips(nCtx: Int) {
        when (nCtx) {
            1024 -> binding.chipGroupContext.check(R.id.chipCtx1024)
            2048 -> binding.chipGroupContext.check(R.id.chipCtx2048)
            4096 -> binding.chipGroupContext.check(R.id.chipCtx4096)
            8192 -> binding.chipGroupContext.check(R.id.chipCtx8192)
            16384 -> binding.chipGroupContext.check(R.id.chipCtx16384)
            else -> binding.chipGroupContext.clearCheck()
        }
    }

    private fun syncBatchChips(nBatch: Int) {
        when (nBatch) {
            128 -> binding.chipGroupBatch.check(R.id.chipBatch128)
            256 -> binding.chipGroupBatch.check(R.id.chipBatch256)
            512 -> binding.chipGroupBatch.check(R.id.chipBatch512)
            1024 -> binding.chipGroupBatch.check(R.id.chipBatch1024)
            else -> binding.chipGroupBatch.check(R.id.chipBatch512)
        }
    }

    private fun readSelectedBatch(): Int {
        return when (binding.chipGroupBatch.checkedChipId) {
            R.id.chipBatch128 -> 128
            R.id.chipBatch256 -> 256
            R.id.chipBatch1024 -> 1024
            else -> 512
        }
    }

    private fun readSelectedPowerMode(): String {
        return when (binding.rgPowerMode.checkedRadioButtonId) {
            R.id.rbPowerBurst -> ModelConfig.POWER_BURST
            R.id.rbPowerBalanced -> ModelConfig.POWER_BALANCED
            R.id.rbPowerSaver -> ModelConfig.POWER_SAVER
            else -> ModelConfig.POWER_SUSTAINED_HIGH
        }
    }

    private fun setupInteractiveListeners() {
        binding.chipCtx1024.setOnClickListener { binding.etMaxContext.setText("1024") }
        binding.chipCtx2048.setOnClickListener { binding.etMaxContext.setText("2048") }
        binding.chipCtx4096.setOnClickListener { binding.etMaxContext.setText("4096") }
        binding.chipCtx8192.setOnClickListener { binding.etMaxContext.setText("8192") }
        binding.chipCtx16384.setOnClickListener { binding.etMaxContext.setText("16384") }

        binding.sliderTemperature.addOnChangeListener { _, value, _ ->
            binding.tvTemperatureVal.text = String.format(Locale.US, "%.2f", value)
        }
        binding.sliderTopP.addOnChangeListener { _, value, _ ->
            binding.tvTopPVal.text = String.format(Locale.US, "%.2f", value)
        }
        binding.sliderMinP.addOnChangeListener { _, value, _ ->
            binding.tvMinPVal.text = String.format(Locale.US, "%.2f", value)
        }
        binding.sliderRepeatPenalty.addOnChangeListener { _, value, _ ->
            binding.tvRepeatPenaltyVal.text = String.format(Locale.US, "%.2f", value)
        }
        binding.sliderThreads.addOnChangeListener { _, value, _ ->
            binding.tvThreadsVal.text = value.roundToInt().toString()
        }

        binding.btnResetDefaults.setOnClickListener {
            populateFromConfig(ModelConfig())
        }

        binding.btnSaveConfig.setOnClickListener {
            val cur = viewModel.uiState.value.config
            val parsedCtx = binding.etMaxContext.text?.toString()?.trim()?.toIntOrNull() ?: cur.nCtx
            val parsedMaxNew = binding.etMaxNewTokens.text?.toString()?.trim()?.toIntOrNull() ?: cur.maxNewTokens
            val parsedTopK = binding.etTopK.text?.toString()?.trim()?.toIntOrNull() ?: cur.topK
            val parsedSeed = binding.etSeed.text?.toString()?.trim()?.toIntOrNull() ?: -1
            val sysPrompt = binding.etSystemPrompt.text?.toString().orEmpty()

            val updated = cur.copy(
                nCtx = parsedCtx,
                maxNewTokens = parsedMaxNew,
                nBatch = readSelectedBatch(),
                nThreads = binding.sliderThreads.value.roundToInt(),
                nNpuLayers = 999, // Строгая привязка всех слоёв к Hexagon NPU
                temperature = binding.sliderTemperature.value,
                topP = binding.sliderTopP.value,
                topK = parsedTopK,
                minP = binding.sliderMinP.value,
                repeatPenalty = binding.sliderRepeatPenalty.value,
                seed = parsedSeed,
                flashAttnMode = if (binding.switchFlashAttn.isChecked) 1 else 0,
                kvCacheQ8 = binding.switchKvQ8.isChecked,
                enableThinking = binding.switchThinkingMode.isChecked,
                powerMode = readSelectedPowerMode(),
                systemPrompt = sysPrompt
            ).sanitized()

            viewModel.updateModelConfig(updated)
            dismiss()
        }
    }

    override fun onDestroyView() {
        _binding = null
        super.onDestroyView()
    }

    companion object {
        const val TAG = "SettingsBottomSheet"
    }
}

package ru.npu.llamachat.model

/**
 * Конфигурация модели и аппаратного ускорителя Qualcomm Hexagon NPU.
 * Все параметры хранятся исключительно в оперативной памяти текущей сессии.
 */
data class ModelConfig(
    val nCtx: Int = 4096,
    val maxNewTokens: Int = 512,
    val nBatch: Int = 512,
    val nThreads: Int = 6,
    /** Всегда >= 1 (999 = выгрузить все слои модели на Hexagon HTP NPU; 0 запрещён). */
    val nNpuLayers: Int = 999,
    val temperature: Float = 0.70f,
    val topP: Float = 0.90f,
    val topK: Int = 40,
    val minP: Float = 0.05f,
    val repeatPenalty: Float = 1.10f,
    val repeatLastN: Int = 64,
    val seed: Int = -1,
    /** 1 = включён (рекомендуется для Hexagon HTP), -1 = авто, 0 = выключен. */
    val flashAttnMode: Int = 1,
    /** true = Q8_0 KV-кэш (экономия ОЗУ), false = F16 KV-кэш. */
    val kvCacheQ8: Boolean = false,
    /** Профиль питания Qualcomm FastRPC cDSP (sustained_high_performance, burst, balanced, power_saver). */
    val powerMode: String = POWER_SUSTAINED_HIGH,
    /** Целевое устройство Hexagon NPU (для Snapdragon 8s Gen 3 v73 строго HTP0). */
    val htpDevices: String = "HTP0",
    val systemPrompt: String = DEFAULT_SYSTEM_PROMPT
) {
    fun sanitized(): ModelConfig {
        val cleanCtx = nCtx.coerceIn(512, 32768)
        val cleanBatch = nBatch.coerceIn(64, 2048).coerceAtMost(cleanCtx)
        val cleanPower = if (powerMode in VALID_POWER_MODES) powerMode else POWER_SUSTAINED_HIGH
        return copy(
            nCtx = cleanCtx,
            maxNewTokens = maxNewTokens.coerceIn(16, 8192),
            nBatch = cleanBatch,
            nThreads = nThreads.coerceIn(1, 8),
            // Строгая политика NPU: никогда не допускаем 0 слоёв на NPU
            nNpuLayers = if (nNpuLayers <= 0 || nNpuLayers > 999) 999 else nNpuLayers,
            temperature = temperature.coerceIn(0.0f, 2.0f),
            topP = topP.coerceIn(0.05f, 1.0f),
            topK = topK.coerceIn(1, 500),
            minP = minP.coerceIn(0.0f, 0.95f),
            repeatPenalty = repeatPenalty.coerceIn(1.0f, 2.5f),
            repeatLastN = repeatLastN.coerceIn(0, 512),
            flashAttnMode = flashAttnMode.coerceIn(-1, 1),
            powerMode = cleanPower,
            htpDevices = htpDevices.trim().ifEmpty { "HTP0" }
        )
    }

    companion object {
        const val POWER_BURST = "burst"
        const val POWER_SUSTAINED_HIGH = "sustained_high_performance"
        const val POWER_HIGH = "high_performance"
        const val POWER_BALANCED = "balanced"
        const val POWER_SAVER = "power_saver"

        val VALID_POWER_MODES = setOf(
            POWER_BURST,
            POWER_SUSTAINED_HIGH,
            POWER_HIGH,
            POWER_BALANCED,
            POWER_SAVER
        )

        val CONTEXT_PRESETS = listOf(1024, 2048, 4096, 8192, 16384)
        val BATCH_PRESETS = listOf(128, 256, 512, 1024)

        const val DEFAULT_SYSTEM_PROMPT =
            "Ты — полезный, точный и лаконичный ИИ-ассистент, работающий локально на нейропроцессоре Qualcomm Hexagon NPU."
    }
}

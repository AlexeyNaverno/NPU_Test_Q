package ru.npu.llamachat.model

/**
 * Сводная телеметрия нейропроцессора Hexagon NPU, счётчиков токенов и оперативной памяти (ОЗУ).
 */
data class LoadedModelInfo(
    val fileName: String,
    val modelDesc: String,
    val quantName: String,
    val fileSizeBytes: Long,
    val paramCount: Long,
    val nCtxTrain: Int,
    val nCtxActive: Int,
    val nLayersNpu: Int,
    val nLayersTotal: Int,
    val htpBufferMib: Double,
    val hostBufferMib: Double,
    val npuDeviceUsed: String,
    val quantWarning: String = ""
)

data class TokenStats(
    val contextUsedTokens: Int = 0,
    val contextMaxTokens: Int = 4096,
    val lastPromptTokens: Int = 0,
    val lastCachedTokens: Int = 0,
    val lastGeneratedTokens: Int = 0,
    val sessionGeneratedTokens: Int = 0,
    val draftInputTokens: Int = 0,
    val prefillTokPerSec: Double = 0.0,
    val decodeTokPerSec: Double = 0.0
)

data class RamStats(
    val processRssMib: Double = 0.0,
    val processPeakRssMib: Double = 0.0,
    val processSwapMib: Double = 0.0,
    val npuHtpBufferMib: Double = 0.0,
    val hostWeightMib: Double = 0.0,
    val systemAvailMib: Double = 0.0,
    val systemTotalMib: Double = 0.0,
    val isLowMemory: Boolean = false
)

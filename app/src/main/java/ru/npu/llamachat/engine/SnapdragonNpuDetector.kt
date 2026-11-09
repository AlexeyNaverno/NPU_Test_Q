package ru.npu.llamachat.engine

import android.os.Build
import java.io.File

/**
 * Детектор платформы Qualcomm Snapdragon и архитектуры нейропроцессора Hexagon HTP.
 * Целевой чипсет по ТЗ: Qualcomm Snapdragon 8s Gen 3 (SM8635, Hexagon HTP v73),
 * а также совместимые платформы Snapdragon с Hexagon HTP NPU.
 */
data class SocNpuReport(
    val socManufacturer: String,
    val socModel: String,
    val marketingName: String,
    val expectedHexagonArch: Int, // Например, 73 для SM8635 (Snapdragon 8s Gen 3)
    val supportsFp16Hmx: Boolean, // false для SM8635 / SM7750 / SM6650 (используется безопасный 4x HVX режим на HTP0)
    val isArm64: Boolean,
    val isQualcommPlatform: Boolean,
    val isSnapdragon8sGen3: Boolean,
    val hasCdspDriverNode: Boolean,
    val blockingReason: String?
)

object SnapdragonNpuDetector {

    private data class KnownSnapdragonSoc(
        val marketingName: String,
        val hexagonArch: Int,
        val supportsFp16Hmx: Boolean
    )

    // Таблица поддерживаемых SoC Qualcomm Snapdragon с Hexagon HTP (cDSP v73..v85).
    // ВАЖНО: На субфлагманских чипах Hexagon v73 (Snapdragon 8s Gen 3 SM8635, 7 Gen 4 SM7750,
    // 7+ Gen 3 SM7675, 6 Gen 4 SM6650, QCM6690) блок FP16 HMX отсутствует либо вызывает
    // переполнение (inf/NaN) на prefill (llama.cpp #31645, #29473, #29740).
    // Для них supportsFp16Hmx = false включает 100% точный векторный режим 4x HVX внутри HTP0 NPU!
    private val KNOWN_SNAPDRAGON_SOCS: Map<String, KnownSnapdragonSoc> = mapOf(
        // Hexagon HTP v73 (HVX-safe NPU mode: без аппаратного FP16 HMX)
        "SM8635"  to KnownSnapdragonSoc("Snapdragon 8s Gen 3", 73, supportsFp16Hmx = false),
        "SM8635P" to KnownSnapdragonSoc("Snapdragon 8s Gen 3", 73, supportsFp16Hmx = false),
        "SM7675"  to KnownSnapdragonSoc("Snapdragon 7+ Gen 3", 73, supportsFp16Hmx = false),
        "SM7675P" to KnownSnapdragonSoc("Snapdragon 7+ Gen 3", 73, supportsFp16Hmx = false),
        "SM7750"  to KnownSnapdragonSoc("Snapdragon 7 Gen 4", 73, supportsFp16Hmx = false),
        "SM7750P" to KnownSnapdragonSoc("Snapdragon 7 Gen 4", 73, supportsFp16Hmx = false),
        "SM6650"  to KnownSnapdragonSoc("Snapdragon 6 Gen 4", 73, supportsFp16Hmx = false),
        "QCM6690" to KnownSnapdragonSoc("Qualcomm Dragonwing Q-6690", 73, supportsFp16Hmx = false),

        // Hexagon HTP v73 (флагманские ядра с полным FP16 HMX)
        "SM8550"  to KnownSnapdragonSoc("Snapdragon 8 Gen 2", 73, supportsFp16Hmx = true),
        "SM8550P" to KnownSnapdragonSoc("Snapdragon 8 Gen 2", 73, supportsFp16Hmx = true),
        "QCS8550" to KnownSnapdragonSoc("Qualcomm QCS8550", 73, supportsFp16Hmx = true),
        "QCS9100" to KnownSnapdragonSoc("Qualcomm IQ-9075", 73, supportsFp16Hmx = true),

        // Hexagon HTP v75
        "SM8650"  to KnownSnapdragonSoc("Snapdragon 8 Gen 3", 75, supportsFp16Hmx = true),
        "SM8650P" to KnownSnapdragonSoc("Snapdragon 8 Gen 3", 75, supportsFp16Hmx = true),
        "QCS8300" to KnownSnapdragonSoc("Qualcomm IQ-8275", 75, supportsFp16Hmx = true),

        // Hexagon HTP v79
        "SM8735"  to KnownSnapdragonSoc("Snapdragon 8s Gen 4", 79, supportsFp16Hmx = true),
        "SM8750"  to KnownSnapdragonSoc("Snapdragon 8 Elite", 79, supportsFp16Hmx = true),
        "SM8750P" to KnownSnapdragonSoc("Snapdragon 8 Elite", 79, supportsFp16Hmx = true),

        // Hexagon HTP v81+
        "SM8850"  to KnownSnapdragonSoc("Snapdragon 8 Elite Gen 2", 81, supportsFp16Hmx = true)
    )

    fun inspectCurrentDevice(): SocNpuReport {
        val rawManufacturer = Build.SOC_MANUFACTURER.orEmpty().trim()
        val rawSocModel = Build.SOC_MODEL.orEmpty().trim().uppercase()
        val hardware = Build.HARDWARE.orEmpty().trim().lowercase()
        val board = Build.BOARD.orEmpty().trim().lowercase()
        val abis = Build.SUPPORTED_ABIS?.toList().orEmpty()

        val isArm64 = abis.any { it.equals("arm64-v8a", ignoreCase = true) }

        val hasCdspDriver = listOf(
            "/vendor/lib64/libcdsprpc.so",
            "/system/vendor/lib64/libcdsprpc.so",
            "/dev/cdsprpc-smd",
            "/dev/adsprpc-smd"
        ).any { path ->
            runCatching { File(path).exists() }.getOrDefault(false)
        }

        return evaluatePlatform(
            rawManufacturer = rawManufacturer,
            rawSocModel = rawSocModel,
            hardware = hardware,
            board = board,
            isArm64 = isArm64,
            hasCdspDriver = hasCdspDriver
        )
    }

    internal fun evaluatePlatform(
        rawManufacturer: String,
        rawSocModel: String,
        hardware: String,
        board: String,
        isArm64: Boolean,
        hasCdspDriver: Boolean
    ): SocNpuReport {
        val normModel = rawSocModel.trim().uppercase()
        val normMfg = rawManufacturer.trim()

        val known = KNOWN_SNAPDRAGON_SOCS[normModel]
            ?: KNOWN_SNAPDRAGON_SOCS.entries.firstOrNull { normModel.startsWith(it.key) }?.value

        val looksQualcommByMfg = normMfg.contains("Qualcomm", ignoreCase = true) ||
            normMfg.equals("QTI", ignoreCase = true)
        val looksQualcommByCode = normModel.startsWith("SM") ||
            normModel.startsWith("QCS") ||
            normModel.startsWith("QCM") ||
            hardware.contains("qcom") ||
            board.contains("pineapple") ||
            board.contains("cliff") || // Кодовое имя платформы SM8635 (Snapdragon 8s Gen 3) = cliff!
            board.contains("kalama") ||
            board.contains("sun")

        val isEmulator = hardware.contains("ranchu") ||
            hardware.contains("goldfish") ||
            board.contains("goldfish") ||
            normModel.contains("EMULATOR", ignoreCase = true)

        val isQualcomm = !isEmulator && (known != null || looksQualcommByMfg || looksQualcommByCode)
        val is8sGen3 = normModel.startsWith("SM8635") || board.contains("cliff")

        val marketing = when {
            known != null -> "${known.marketingName} ($normModel)"
            normModel.isNotEmpty() -> "$normMfg $normModel".trim()
            else -> "Неизвестный SoC ($hardware)"
        }

        val blockingReason = when {
            !isArm64 ->
                "Архитектура процессора не является arm64-v8a. Модули Qualcomm Hexagon HTP требуют 64-битной ARM-платформы."
            isEmulator ->
                "Обнаружен Android-эмулятор ($hardware). Эмуляция нейропроцессора Qualcomm Hexagon NPU невозможна, а фоллбэк на CPU/GPU запрещён."
            !isQualcomm ->
                "Обнаружен не-Snapdragon чипсет ($marketing). Приложение предназначено строго для NPU Qualcomm Hexagon (Snapdragon 8s Gen 3 и совместимых); запуск на CPU/GPU заблокирован."
            else -> null
        }

        // Для любых неизвестных платформ или SM8635 (cliff) по умолчанию выбираем безопасный HVX-режим (false),
        // чтобы гарантированно исключить порчу логитов из-за отсутствующего FP16 HMX.
        val fp16HmxSafe = if (is8sGen3) false else (known?.supportsFp16Hmx ?: false)

        return SocNpuReport(
            socManufacturer = normMfg.ifEmpty { "Unknown" },
            socModel = normModel.ifEmpty { hardware.uppercase() },
            marketingName = marketing,
            expectedHexagonArch = known?.hexagonArch ?: 0,
            supportsFp16Hmx = fp16HmxSafe,
            isArm64 = isArm64,
            isQualcommPlatform = isQualcomm,
            isSnapdragon8sGen3 = is8sGen3,
            hasCdspDriverNode = hasCdspDriver,
            blockingReason = blockingReason
        )
    }
}

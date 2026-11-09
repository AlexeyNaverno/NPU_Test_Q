package ru.npu.llamachat.engine

import java.util.Locale

/**
 * Двухконтурный предварительный контроль вместимости Qualcomm Hexagon NPU и системного ОЗУ
 * (Этап 1 — на уровне SAF/Kotlin ДО открытия нативного загрузчика весов и выделения DMA-BUF).
 *
 * Предотвращает аварийное завершение приложения (SIGKILL от Android lmkd / SIGABRT от fastrpc_mmap)
 * при выборе слишком больших моделей (например, 7B / 9B / 14B .gguf).
 */
data class NpuBudgetPrecheckResult(
    val allowed: Boolean,
    val errorMessage: String = ""
)

object NpuMemoryBudgetEvaluator {

    private const val BYTES_PER_MIB = 1024.0 * 1024.0

    /** Минимальный корректный размер GGUF файла (заголовок + метаданные + тензор). */
    private const val MIN_VALID_GGUF_BYTES = 1024L

    /** Безопасный невыгружаемый резерв ОЗУ для ОС Android, чтобы lmkd не убил процесс при rpcmem_alloc2. */
    private const val ANDROID_LMKD_HEADROOM_MIB = 700.0

    /** Минимальный запас на KV-кэш + буфер активаций HTP поверх размера файла весов. */
    private const val MIN_RUNTIME_NPU_OVERHEAD_MIB = 320.0

    fun checkBeforeNativeLoad(
        fileName: String,
        fileSizeBytes: Long,
        hexagonArch: Int,
        activeHtpDeviceCount: Int,
        systemAvailMib: Double,
        systemTotalMib: Double,
        reclaimableLoadedModelMib: Double = 0.0
    ): NpuBudgetPrecheckResult {
        if (fileSizeBytes in 1 until MIN_VALID_GGUF_BYTES) {
            return NpuBudgetPrecheckResult(
                allowed = false,
                errorMessage = "Файл «$fileName» ($fileSizeBytes байт) повреждён или пуст и не является корректной моделью .gguf."
            )
        }

        // Если провайдер документов не вернул размер (<= 0), пропускаем в Этап 2 (точный C++ GGUF-инспектор).
        if (fileSizeBytes <= 0L) {
            return NpuBudgetPrecheckResult(allowed = true)
        }

        val fileSizeMib = fileSizeBytes.toDouble() / BYTES_PER_MIB
        val arch = if (hexagonArch > 0) hexagonArch else 73
        val ndev = if (arch < 75) 1 else activeHtpDeviceCount.coerceAtLeast(1)

        // 1. Проверка потолка адресного пространства Hexagon cDSP (32-бит VA на v73..v79: 3000/3200 МБ всего)
        val (vmemCeilingMib, maxWeightFileMib) = if (arch <= 79) {
            val perDevVmem = if (arch >= 75) 3200.0 else 3000.0
            val totalVmem = perDevVmem * ndev
            val weightCap = (perDevVmem - 500.0) * ndev // ~2500 МБ на v73, ~2700 МБ на v75/v79
            totalVmem to weightCap
        } else {
            val sysCap = if (systemTotalMib > 0.0) systemTotalMib * 0.70 else 8192.0
            sysCap to (sysCap - 500.0).coerceAtLeast(1024.0)
        }

        if (fileSizeMib > maxWeightFileMib) {
            val msg = buildString {
                append("Модель «$fileName» слишком велика для размещения в Qualcomm Hexagon NPU.\n\n")
                append("• Размер выбранного файла: ${formatMibOrGib(fileSizeMib)}\n")
                append("• Аппаратный лимит весов Hexagon HTP v$arch: ${formatMibOrGib(maxWeightFileMib)} ")
                if (arch <= 79) {
                    append("(32-битное адресное пространство cDSP ограничено ${vmemCeilingMib.toInt()} МБ)\n\n")
                } else {
                    append("\n\n")
                }
                append("Загрузка остановлена до выделения памяти — вылет приложения предотвращён.\n")
                append("Выберите модель размера 0.5B–3B (до 4B в чистом Q4_0) с квантованием Q4_0 (int4) или Q8_0.")
            }
            return NpuBudgetPrecheckResult(allowed = false, errorMessage = msg)
        }

        // 2. Проверка доступного физического ОЗУ Android (защита от SIGKILL со стороны lmkd при rpcmem_alloc2)
        val effectiveAvailMib = if (systemAvailMib > 0.0) {
            systemAvailMib + reclaimableLoadedModelMib.coerceAtLeast(0.0)
        } else {
            0.0
        }

        val minRequiredRamMib = fileSizeMib + MIN_RUNTIME_NPU_OVERHEAD_MIB
        if (effectiveAvailMib > 0.0 && (minRequiredRamMib + ANDROID_LMKD_HEADROOM_MIB) > effectiveAvailMib) {
            val msg = buildString {
                append("Недостаточно свободного ОЗУ для безопасной загрузки «$fileName» в буферы Hexagon NPU (DMA-BUF).\n\n")
                append("• Минимум для весов и буферов NPU: ~${formatMibOrGib(minRequiredRamMib)}\n")
                append("• Необходимый резерв ОС Android: ${ANDROID_LMKD_HEADROOM_MIB.toInt()} МБ\n")
                append("• Доступно ОЗУ сейчас: ${formatMibOrGib(effectiveAvailMib)}\n\n")
                append("Выгрузка на NPU отменена до выделения памяти во избежание аварийного закрытия приложения системой. ")
                append("Закройте фоновые приложения или выберите компактную модель (0.5B–1.7B Q4_0).")
            }
            return NpuBudgetPrecheckResult(allowed = false, errorMessage = msg)
        }

        return NpuBudgetPrecheckResult(allowed = true)
    }

    private fun formatMibOrGib(mib: Double): String {
        return if (mib >= 1024.0) {
            String.format(Locale.US, "%.2f ГБ (%.0f МБ)", mib / 1024.0, mib)
        } else {
            String.format(Locale.US, "%.0f МБ", mib)
        }
    }
}

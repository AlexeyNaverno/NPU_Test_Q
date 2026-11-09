package ru.npu.llamachat.engine

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class NpuMemoryBudgetEvaluatorTest {

    private val mib = 1024L * 1024L

    @Test
    fun `rejects oversized 9B model on Hexagon HTP v73 even with 16 GB system RAM`() {
        val res = NpuMemoryBudgetEvaluator.checkBeforeNativeLoad(
            fileName = "Qwen3.5-9B-Q4_0.gguf",
            fileSizeBytes = 5400L * mib, // ~5.27 GiB
            hexagonArch = 73,
            activeHtpDeviceCount = 1,
            systemAvailMib = 11200.0,
            systemTotalMib = 16384.0
        )
        assertFalse(res.allowed)
        assertTrue(res.errorMessage.contains("слишком велика"))
        assertTrue(res.errorMessage.contains("Q4_0"))
    }

    @Test
    fun `rejects model when free Android RAM would trigger lmkd SIGKILL`() {
        val tightRam = NpuMemoryBudgetEvaluator.checkBeforeNativeLoad(
            fileName = "Llama-3.2-3B-Q4_0.gguf",
            fileSizeBytes = 1900L * mib,
            hexagonArch = 73,
            activeHtpDeviceCount = 1,
            systemAvailMib = 2200.0, // 1900 + 320 + 700 = 2920 > 2200
            systemTotalMib = 8192.0,
            reclaimableLoadedModelMib = 0.0
        )
        assertFalse(tightRam.allowed)
        assertTrue(tightRam.errorMessage.contains("Недостаточно свободного ОЗУ"))

        // If 1000 MiB is reclaimable from an already-loaded model, 2200 + 1000 = 3200 >= 2920 -> allowed!
        val withReclaim = NpuMemoryBudgetEvaluator.checkBeforeNativeLoad(
            fileName = "Llama-3.2-3B-Q4_0.gguf",
            fileSizeBytes = 1900L * mib,
            hexagonArch = 73,
            activeHtpDeviceCount = 1,
            systemAvailMib = 2200.0,
            systemTotalMib = 8192.0,
            reclaimableLoadedModelMib = 1000.0
        )
        assertTrue(withReclaim.allowed)
    }

    @Test
    fun `accepts Qwen3 0_6B Q4_0 and 1_7B Q4_0 on Snapdragon 8s Gen 3 v73`() {
        val qwen06b = NpuMemoryBudgetEvaluator.checkBeforeNativeLoad(
            fileName = "Qwen3-0.6B-Q4_0.gguf",
            fileSizeBytes = 415L * mib,
            hexagonArch = 73,
            activeHtpDeviceCount = 1,
            systemAvailMib = 4200.0,
            systemTotalMib = 12288.0
        )
        assertTrue(qwen06b.allowed)

        val qwen17b = NpuMemoryBudgetEvaluator.checkBeforeNativeLoad(
            fileName = "Qwen3-1.7B-Q4_0.gguf",
            fileSizeBytes = 1120L * mib,
            hexagonArch = 73,
            activeHtpDeviceCount = 1,
            systemAvailMib = 4200.0,
            systemTotalMib = 12288.0
        )
        assertTrue(qwen17b.allowed)
    }

    @Test
    fun `rejects tiny corrupted GGUF stub file`() {
        val corrupt = NpuMemoryBudgetEvaluator.checkBeforeNativeLoad(
            fileName = "broken.gguf",
            fileSizeBytes = 64L,
            hexagonArch = 73,
            activeHtpDeviceCount = 1,
            systemAvailMib = 6000.0,
            systemTotalMib = 12288.0
        )
        assertFalse(corrupt.allowed)
    }
}

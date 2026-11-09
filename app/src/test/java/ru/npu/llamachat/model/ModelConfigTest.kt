package ru.npu.llamachat.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelConfigTest {

    @Test
    fun `sanitized never allows zero NPU layers`() {
        val zeroAttempt = ModelConfig(nNpuLayers = 0).sanitized()
        assertEquals(999, zeroAttempt.nNpuLayers)

        val negativeAttempt = ModelConfig(nNpuLayers = -5).sanitized()
        assertEquals(999, negativeAttempt.nNpuLayers)
    }

    @Test
    fun `sanitized clamps context batch and sampling bounds`() {
        val cfg = ModelConfig(
            nCtx = 128,
            nBatch = 4096,
            maxNewTokens = 2,
            nThreads = 99,
            temperature = 9.5f,
            topP = 0.0f,
            repeatPenalty = 0.2f,
            powerMode = "unsupported_profile",
            htpDevices = "   "
        ).sanitized()

        assertEquals(512, cfg.nCtx)
        assertTrue(cfg.nBatch <= cfg.nCtx)
        assertEquals(16, cfg.maxNewTokens)
        assertEquals(8, cfg.nThreads)
        assertEquals(2.0f, cfg.temperature, 1e-4f)
        assertEquals(0.05f, cfg.topP, 1e-4f)
        assertEquals(1.0f, cfg.repeatPenalty, 1e-4f)
        assertEquals(ModelConfig.POWER_SUSTAINED_HIGH, cfg.powerMode)
        assertEquals("HTP0", cfg.htpDevices)
    }
}

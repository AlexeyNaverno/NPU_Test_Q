package ru.npu.llamachat.image

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class LcmSchedulerTest {
    @Test fun `four and five step schedules match LCM reference`() {
        assertEquals(listOf(999, 759, 499, 259), LcmScheduler(4).timesteps)
        assertEquals(listOf(999, 799, 599, 399, 199), LcmScheduler(5).timesteps)
    }

    @Test fun `step uses provided noise and leaves inputs intact`() {
        val scheduler = LcmScheduler(4)
        val sample = floatArrayOf(0.5f)
        val prediction = floatArrayOf(0.1f)
        val noise = floatArrayOf(0.25f)
        assertEquals(1.58307699f, scheduler.step(0, sample, prediction, noise)[0], 1e-5f)
        assertArrayEquals(floatArrayOf(0.5f), sample, 0f)
        assertArrayEquals(floatArrayOf(0.1f), prediction, 0f)
        assertArrayEquals(floatArrayOf(0.25f), noise, 0f)
    }

    @Test fun `last step is denoised output and ignores noise`() {
        val scheduler = LcmScheduler(4)
        val x = floatArrayOf(0.5f)
        val eps = floatArrayOf(0.1f)
        val result = scheduler.step(3, x, eps)
        assertEquals(0.54399772f, result[0], 1e-5f)
        assertArrayEquals(result, scheduler.step(3, x, eps, floatArrayOf(100f)), 0f)
    }

    @Test fun `invalid dimensions and schedules are rejected`() {
        assertThrows(IllegalArgumentException::class.java) { LcmScheduler(0) }
        assertThrows(IllegalArgumentException::class.java) { LcmScheduler(51) }
        val scheduler = LcmScheduler(4)
        assertThrows(IllegalArgumentException::class.java) { scheduler.step(0, floatArrayOf(1f), floatArrayOf(1f)) }
        assertThrows(IllegalArgumentException::class.java) { scheduler.step(0, floatArrayOf(1f), floatArrayOf(), floatArrayOf(1f)) }
        assertThrows(IllegalArgumentException::class.java) { scheduler.step(4, floatArrayOf(1f), floatArrayOf(1f)) }
    }
}

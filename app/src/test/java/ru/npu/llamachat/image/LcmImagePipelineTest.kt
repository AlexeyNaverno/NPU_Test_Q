package ru.npu.llamachat.image

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import java.util.concurrent.CancellationException

class LcmImagePipelineTest {
    @Test fun `calls stages in order with four LCM timesteps and maps decoded output`() {
        val calls = mutableListOf<String>()
        val pipeline = LcmImagePipeline(
            { prompt -> calls.add("clip:$prompt"); floatArrayOf(42f) },
            { latent, t, embedding ->
                calls.add("unet:$t")
                assertArrayEquals(floatArrayOf(42f), embedding, 0f)
                FloatArray(latent.size)
            },
            { latent -> calls.add("vae"); assertEquals(2, latent.size); floatArrayOf(-2f, 0f, 2f) },
            latentSize = 2, imageSize = 3
        )
        val progress = mutableListOf<Int>()
        val result = pipeline.generate("кот", seed = 123, onStep = { current, total ->
            assertEquals(4, total); progress.add(current)
        })
        assertEquals(listOf("clip:кот", "unet:999", "unet:759", "unet:499", "unet:259", "vae"), calls)
        assertEquals(listOf(1, 2, 3, 4), progress)
        assertArrayEquals(floatArrayOf(0f, 0.5f, 1f), result, 0f)
    }

    @Test fun `seed is reproducible and decoder receives scaled latent`() {
        val pipeline = LcmImagePipeline(
            { floatArrayOf(1f) }, { latent, _, _ -> FloatArray(latent.size) },
            { it }, latentSize = 4, imageSize = 4, latentScale = 0.5f
        )
        val a = pipeline.generate("test", steps = 1, seed = 7)
        assertArrayEquals(a, pipeline.generate("test", steps = 1, seed = 7), 0f)
        val scheduler = LcmScheduler(1)
        val random = java.util.Random(7)
        val initial = FloatArray(4) { random.nextGaussian().toFloat() }
        val denoised = scheduler.step(0, initial, FloatArray(4))
        val expected = FloatArray(4) { ((denoised[it] / 0.5f + 1f) / 2f).coerceIn(0f, 1f) }
        assertArrayEquals(expected, a, 0f)
    }

    @Test fun `decoder output already in zero to one range is not rescaled`() {
        val pipeline = LcmImagePipeline(
            { floatArrayOf(1f) }, { latent, _, _ -> FloatArray(latent.size) },
            { floatArrayOf(-1f, 0.25f, 2f) }, latentSize = 1, imageSize = 3,
            decoderRange = LcmImagePipeline.DecoderRange.ZERO_TO_ONE
        )
        assertArrayEquals(floatArrayOf(0f, 0.25f, 1f), pipeline.generate("test", steps = 1, seed = 0), 0f)
    }

    @Test fun `invalid output and cancellation prevent decoding`() {
        var decoded = false
        val bad = LcmImagePipeline(
            { floatArrayOf(1f) }, { _, _, _ -> floatArrayOf(Float.NaN) },
            { decoded = true; floatArrayOf(0f) }, latentSize = 1, imageSize = 1
        )
        assertThrows(IllegalArgumentException::class.java) { bad.generate("test", seed = 0) }
        assertEquals(false, decoded)
        val cancelled = LcmImagePipeline(
            { floatArrayOf(1f) }, { _, _, _ -> throw CancellationException() },
            { decoded = true; floatArrayOf(0f) }, latentSize = 1, imageSize = 1
        )
        assertThrows(CancellationException::class.java) { cancelled.generate("test", seed = 0) }
        assertEquals(false, decoded)
    }
}

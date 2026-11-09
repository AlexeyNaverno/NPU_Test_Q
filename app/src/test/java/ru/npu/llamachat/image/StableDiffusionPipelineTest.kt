package ru.npu.llamachat.image

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class StableDiffusionPipelineTest {
    @Test fun `DDIM leading spacing and numerical step match reference`() {
        val scheduler = DdimScheduler(20)
        assertEquals(951, scheduler.timesteps.first())
        assertEquals(1, scheduler.timesteps.last())
        assertEquals(0.62402165f, scheduler.step(0, floatArrayOf(0.5f), floatArrayOf(0.1f))[0], 1e-5f)
        assertThrows(IllegalArgumentException::class.java) { DdimScheduler(101) }
    }

    @Test fun `uses negative and positive embeddings with two UNet passes and converts CHW`() {
        val calls = mutableListOf<String>()
        val pipeline = StableDiffusionPipeline(
            { text -> calls.add("clip:$text"); floatArrayOf(if (text == "cat") 2f else 1f) },
            { latent, time, embedding -> calls.add("unet:$time:${embedding[0]}"); FloatArray(latent.size) },
            { calls.add("vae"); floatArrayOf(-1f, 1f, 0f, 0f, 1f, -1f) },
            latentSize = 2, outputPixels = 2
        )
        val rgb = pipeline.generate("cat", "bad", 2, 7.5f, 1)
        assertEquals(listOf("clip:cat", "clip:bad", "unet:501:2.0", "unet:501:1.0", "unet:1:2.0", "unet:1:1.0", "vae"), calls)
        assertArrayEquals(floatArrayOf(0f, 0.5f, 1f, 1f, 0.5f, 0f), rgb, 0f)
    }

    @Test fun `guidance one skips negative pass and invalid values fail`() {
        var clips = 0
        var unets = 0
        val pipeline = StableDiffusionPipeline({ clips++; floatArrayOf(1f) },
            { latent, _, _ -> unets++; FloatArray(latent.size) },
            { floatArrayOf(0f, 0f, 0f) }, latentSize = 1, outputPixels = 1)
        pipeline.generate("x", "", 1, 1f, 0)
        assertEquals(1, clips)
        assertEquals(1, unets)
        assertThrows(IllegalArgumentException::class.java) { pipeline.generate("x", "", 1, Float.NaN, 0) }
        assertThrows(IllegalArgumentException::class.java) { pipeline.generate("x", "", 0, 1f, 0) }
    }
}

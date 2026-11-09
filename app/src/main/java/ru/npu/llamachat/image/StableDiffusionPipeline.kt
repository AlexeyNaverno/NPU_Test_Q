package ru.npu.llamachat.image

import java.util.Random

/** Классический SD 1.5: DDIM eta=0, epsilon prediction, CFG, латент NCHW. */
internal class StableDiffusionPipeline(
    private val textEncoder: (String) -> FloatArray,
    private val unet: (FloatArray, Int, FloatArray) -> FloatArray,
    private val vae: (FloatArray) -> FloatArray,
    private val latentSize: Int = 4 * 64 * 64,
    private val outputPixels: Int = 512 * 512
) {
    fun generate(prompt: String, negativePrompt: String, steps: Int, guidance: Float, seed: Long,
                 checkCancelled: () -> Unit = {}, onStep: (Int, Int) -> Unit = { _, _ -> }): FloatArray {
        require(prompt.isNotBlank() && prompt.length <= 4096 && negativePrompt.length <= 4096)
        require(latentSize > 0 && outputPixels > 0 && guidance.isFinite() && guidance in 1f..20f)
        val scheduler = DdimScheduler(steps)
        checkCancelled()
        val positive = textEncoder(prompt)
        require(positive.isNotEmpty() && positive.all { it.isFinite() })
        checkCancelled()
        val negative = if (guidance != 1f) textEncoder(negativePrompt) else null
        require(negative == null || negative.size == positive.size && negative.all { it.isFinite() })
        val random = Random(seed)
        var latent = FloatArray(latentSize) { random.nextGaussian().toFloat() }
        for ((index, t) in scheduler.timesteps.withIndex()) {
            checkCancelled()
            val cond = unet(latent, t, positive)
            require(cond.size == latentSize && cond.all { it.isFinite() }) { "Неверный выход UNet" }
            val prediction = if (negative == null) cond else {
                checkCancelled()
                val uncond = unet(latent, t, negative)
                require(uncond.size == latentSize && uncond.all { it.isFinite() }) { "Неверный выход UNet" }
                FloatArray(latentSize) { uncond[it] + guidance * (cond[it] - uncond[it]) }
            }
            checkCancelled()
            latent = scheduler.step(index, latent, prediction)
            require(latent.all { it.isFinite() }) { "Численное переполнение DDIM" }
            onStep(index + 1, steps)
        }
        checkCancelled()
        val decoded = vae(FloatArray(latentSize) { latent[it] / 0.18215f })
        require(decoded.size == 3 * outputPixels && decoded.all { it.isFinite() }) { "Неверный выход VAE" }
        checkCancelled()
        // VAE выводит NCHW в диапазоне [-1,1], UI ожидает RGB NHWC [0,1].
        return FloatArray(3 * outputPixels) { i -> ((decoded[(i % 3) * outputPixels + i / 3] + 1f) / 2f).coerceIn(0f, 1f) }
    }
}

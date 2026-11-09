package ru.npu.llamachat.image

import java.util.Random

/**
 * Экспериментальная оркестрация SD 1.5 LCM, batch=1, CFG=1.
 * Адаптеры отвечают за токенизацию, преобразование типов, форму и выполнение графов.
 * Здесь нет реализации модели: конкретный адаптер UNet/VAE обязан проверять HTP и
 * запрещать fallback. Нельзя считать наличие этого класса работающим генератором.
 */
internal class LcmImagePipeline(
    private val textEncoder: TextEncoder,
    private val unet: Denoiser,
    private val vae: Decoder,
    private val latentSize: Int = 4 * 64 * 64,
    private val imageSize: Int = 3 * 512 * 512,
    private val latentScale: Float = 0.18215f,
    private val decoderRange: DecoderRange = DecoderRange.NEGATIVE_ONE_TO_ONE
) {
    enum class DecoderRange { NEGATIVE_ONE_TO_ONE, ZERO_TO_ONE }

    fun interface TextEncoder { fun encode(prompt: String): FloatArray }
    fun interface Denoiser { fun predict(latent: FloatArray, timestep: Int, embedding: FloatArray): FloatArray }
    fun interface Decoder { fun decode(latent: FloatArray): FloatArray }

    init {
        require(latentSize > 0 && imageSize > 0)
        require(latentScale.isFinite() && latentScale > 0f)
    }

    /** Возвращает значения в раскладке декодера и диапазоне [0, 1], без сохранения на диск. */
    fun generate(
        prompt: String,
        steps: Int = 4,
        seed: Long,
        checkCancelled: () -> Unit = {},
        onStep: (Int, Int) -> Unit = { _, _ -> }
    ): FloatArray {
        require(prompt.isNotBlank()) { "Введите описание изображения" }
        val scheduler = LcmScheduler(steps)
        val random = Random(seed)
        fun noise() = FloatArray(latentSize) { random.nextGaussian().toFloat() }
        checkCancelled()
        val embedding = textEncoder.encode(prompt)
        require(embedding.isNotEmpty() && embedding.all { it.isFinite() }) { "Некорректный результат CLIP" }
        var latent = noise()
        scheduler.timesteps.forEachIndexed { index, timestep ->
            checkCancelled()
            val prediction = unet.predict(latent, timestep, embedding)
            require(prediction.size == latentSize && prediction.all { it.isFinite() }) {
                "Некорректный результат UNet"
            }
            checkCancelled()
            latent = scheduler.step(index, latent, prediction, if (index == steps - 1) null else noise())
            require(latent.all { it.isFinite() }) { "Некорректный результат LCM" }
            onStep(index + 1, steps)
        }
        checkCancelled()
        val decoded = vae.decode(FloatArray(latentSize) { latent[it] / latentScale })
        require(decoded.size == imageSize && decoded.all { it.isFinite() }) { "Некорректный результат VAE" }
        checkCancelled()
        return FloatArray(imageSize) {
            val value = when (decoderRange) {
                DecoderRange.NEGATIVE_ONE_TO_ONE -> (decoded[it] + 1f) / 2f
                DecoderRange.ZERO_TO_ONE -> decoded[it]
            }
            value.coerceIn(0f, 1f)
        }
    }
}

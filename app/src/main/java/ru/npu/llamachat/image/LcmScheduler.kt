package ru.npu.llamachat.image

import kotlin.math.sqrt

/**
 * Математика LCM для SD 1.5 с epsilon-prediction и scaled-linear beta schedule.
 * Это только сэмплер: он не загружает модели и не выполняет нейросетевые вычисления.
 * Формулы соответствуют LCMScheduler из Hugging Face diffusers. Требуется checkpoint,
 * обученный или адаптированный под LCM, с соответствующей конфигурацией scheduler.
 */
class LcmScheduler(
    val steps: Int,
    private val trainingSteps: Int = 1000,
    private val originalSteps: Int = 50,
    betaStart: Double = 0.00085,
    betaEnd: Double = 0.012,
    private val timestepScaling: Double = 10.0
) {
    init {
        require(trainingSteps > 1 && originalSteps in 1..trainingSteps)
        require(steps in 1..originalSteps)
        require(betaStart > 0.0 && betaEnd >= betaStart && betaEnd < 1.0)
        require(timestepScaling.isFinite() && timestepScaling > 0.0)
    }

    private val alphas: DoubleArray = DoubleArray(trainingSteps).also { values ->
        val start = sqrt(betaStart)
        val end = sqrt(betaEnd)
        var cumulative = 1.0
        for (i in values.indices) {
            val value = start + (end - start) * i / (trainingSteps - 1)
            cumulative *= 1.0 - value * value
            values[i] = cumulative
        }
    }

    /** Временные метки обучения, в порядке убывания. */
    val timesteps: List<Int> = List(steps) { index ->
        val originalIndex = (index.toLong() * originalSteps / steps).toInt()
        (originalSteps - originalIndex) * (trainingSteps / originalSteps) - 1
    }

    /**
     * Один шаг: шум задаётся вызывающим кодом для воспроизводимости; на последнем шаге
     * шум не используется. Исходные массивы не изменяются.
     */
    fun step(index: Int, sample: FloatArray, modelOutput: FloatArray, noise: FloatArray? = null): FloatArray {
        require(index in timesteps.indices)
        require(sample.isNotEmpty() && modelOutput.size == sample.size)
        if (index != steps - 1) require(noise != null && noise.size == sample.size)
        else require(noise == null || noise.size == sample.size)

        val t = timesteps[index]
        val alpha = alphas[t]
        val sqrtAlpha = sqrt(alpha)
        val sqrtBeta = sqrt(1.0 - alpha)
        val scaled = t * timestepScaling
        val cSkip = 0.25 / (scaled * scaled + 0.25)
        val cOut = scaled / sqrt(scaled * scaled + 0.25)
        val last = index == steps - 1
        val previousAlpha = if (last) 1.0 else alphas[timesteps[index + 1]]
        val sqrtPreviousAlpha = sqrt(previousAlpha)
        val sqrtPreviousBeta = sqrt(1.0 - previousAlpha)
        return FloatArray(sample.size) { i ->
            val predictedOriginal = (sample[i] - sqrtBeta * modelOutput[i]) / sqrtAlpha
            val denoised = cOut * predictedOriginal + cSkip * sample[i]
            (if (last) denoised else sqrtPreviousAlpha * denoised + sqrtPreviousBeta * noise!![i]).toFloat()
        }
    }
}

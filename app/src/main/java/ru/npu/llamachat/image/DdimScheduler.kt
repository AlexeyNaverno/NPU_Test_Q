package ru.npu.llamachat.image

import kotlin.math.sqrt

/** SD 1.5 DDIM, epsilon prediction, eta=0 и scaled-linear beta schedule. */
internal class DdimScheduler(val steps: Int) {
    init { require(steps in 1..100) { "Число шагов должно быть от 1 до 100" } }
    private val alphas = DoubleArray(1000).also { values ->
        var cumulative = 1.0
        for (i in values.indices) {
            val betaRoot = sqrt(0.00085) + (sqrt(0.012) - sqrt(0.00085)) * i / 999
            cumulative *= 1.0 - betaRoot * betaRoot
            values[i] = cumulative
        }
    }
    // Diffusers DDIM leading spacing, steps_offset=1, set_alpha_to_one=false.
    private val ratio = 1000 / steps
    val timesteps = List(steps) { (steps - 1 - it) * ratio + 1 }

    fun step(index: Int, sample: FloatArray, epsilon: FloatArray): FloatArray {
        require(index in timesteps.indices && sample.isNotEmpty() && sample.size == epsilon.size)
        val t = timesteps[index]
        val previousT = t - ratio
        val alpha = alphas[t]
        val previousAlpha = if (previousT < 0) alphas[0] else alphas[previousT]
        return FloatArray(sample.size) { i ->
            val predicted = (sample[i] - sqrt(1.0 - alpha) * epsilon[i]) / sqrt(alpha)
            (sqrt(previousAlpha) * predicted + sqrt(1.0 - previousAlpha) * epsilon[i]).toFloat()
        }
    }
}

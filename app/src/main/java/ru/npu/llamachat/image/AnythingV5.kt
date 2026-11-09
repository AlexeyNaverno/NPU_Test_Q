package ru.npu.llamachat.image

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Пробный адаптер обычного ONNX Anything V5. CLIP — CPU; UNet и VAE открываются
 * исключительно с QNN HTP и без CPU fallback. Модель динамически квантована,
 * поэтому создание сессии на многих устройствах может завершиться ошибкой.
 */
internal object AnythingV5 {
    fun generate(context: Context, directory: File, prompt: String, negativePrompt: String,
                 steps: Int, guidance: Float, seed: Long, checkCancelled: () -> Unit,
                 onStep: (Int, Int) -> Unit): FloatArray {
        val tokenizer = ClipTokenizerFiles.load(File(directory, "tokenizer"))
        val env = OrtEnvironment.getEnvironment()
        var clip: OrtSession? = null
        var unet: QnnHtpRuntime.HtpSession? = null
        try {
            val pipeline = StableDiffusionPipeline(
                textEncoder = { text ->
                    val session = clip ?: env.createSession(File(directory, "text_encoder/model.onnx").absolutePath).also { clip = it }
                    val tokens = tokenizer.tokenize(text).map { it.toLong() }.toLongArray()
                    OnnxTensor.createTensor(env, LongBuffer.wrap(tokens), longArrayOf(1, 77)).use { input ->
                        session.run(mapOf("input_ids" to input), setOf("last_hidden_state")).use { result ->
                            floats(result, "last_hidden_state", 77 * 768)
                        }
                    }
                },
                unet = { latent, timestep, embedding ->
                    clip?.close(); clip = null
                    val model = unet ?: QnnHtpRuntime.open(context, File(directory, "unet/model.onnx"),
                        symbolicDimensions = mapOf("batch_size" to 1L, "height" to 64L, "width" to 64L,
                            "sequence_length" to 77L)).also { unet = it }
                    OnnxTensor.createTensor(env, FloatBuffer.wrap(latent), longArrayOf(1, 4, 64, 64)).use { sample ->
                        OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(timestep.toFloat())), longArrayOf()).use { time ->
                            OnnxTensor.createTensor(env, FloatBuffer.wrap(embedding), longArrayOf(1, 77, 768)).use { text ->
                                model.session.run(mapOf("sample" to sample, "timestep" to time, "encoder_hidden_states" to text),
                                    setOf("out_sample")).use { floats(it, "out_sample", 4 * 64 * 64) }
                            }
                        }
                    }
                },
                vae = { latent ->
                    unet?.close(); unet = null
                    checkCancelled()
                    QnnHtpRuntime.open(context, File(directory, "vae_decoder/model.onnx"),
                        symbolicDimensions = mapOf("batch_size" to 1L, "latent_height" to 64L, "latent_width" to 64L)).use { model ->
                        OnnxTensor.createTensor(env, FloatBuffer.wrap(latent), longArrayOf(1, 4, 64, 64)).use { sample ->
                            model.session.run(mapOf("latent_sample" to sample), setOf("sample")).use {
                                floats(it, "sample", 3 * 512 * 512)
                            }
                        }
                    }
                }
            )
            return pipeline.generate(prompt, negativePrompt, steps, guidance, seed, checkCancelled, onStep)
        } finally {
            try { clip?.close() } finally { unet?.close() }
        }
    }

    private fun floats(result: OrtSession.Result, name: String, expected: Int): FloatArray {
        val value = result.get(name).orElseThrow { IllegalStateException("Выход $name отсутствует") } as? OnnxTensor
            ?: error("Выход $name не является тензором")
        val buffer = value.floatBuffer ?: error("Выход $name не имеет тип FLOAT")
        require(buffer.remaining() == expected) { "Неверный размер выхода $name" }
        return FloatArray(expected).also { buffer.get(it) }
    }
}

package ru.npu.llamachat.image

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.TensorInfo
import android.content.Context
import java.io.File
import java.nio.IntBuffer

/**
 * Экспериментальный pipeline для конкретного DreamShaper8 LCM 8gen2, ревизия
 * 6a2e524fe1e74f8050ea169f6cec1f725f956d90. Все три графа требуют QNN HTP;
 * любые неподдерживаемые операторы приводят к ошибке, без CPU fallback.
 * Файлы контекста должны находиться рядом с ONNX. До вызова необходимо освободить
 * LLM и обеспечить эксклюзивный доступ к NPU. На реальном устройстве не проверен.
 */
internal object DreamShaperLcm {
    private val latentShape = listOf(1L, 64L, 64L, 4L) // NHWC
    private val textShape = listOf(1L, 77L, 768L)
    private val imageShape = listOf(1L, 512L, 512L, 3L) // NHWC
    private fun spec(name: String, shape: List<Long>, scale: Double, zero: Int) =
        QuantizedTensorSpec(name, shape, AffineUInt16Quantizer(scale, zero))
    private val unetTime = spec("timestep", listOf(1, 1), 0.012222476303577423, 0)
    private val unetLatent = spec("latent", latentShape, 0.0003101392067037523, 38215)
    private val unetText = spec("text_emb", textShape, 0.0009330504108220339, 30128)
    private val unetOutput = spec("output_latent", latentShape, 0.00020076349028386176, 37133)
    private val vaeInput = spec("latent", latentShape, 0.0005078919348306954, 30969)
    private val vaeOutput = spec("image", imageShape, 1.5259021893143654e-05, 0)
    private val textOutput = spec("text_embedding", textShape, 0.000931622285861522, 30073)

    /** Возвращает RGB NHWC [512,512,3] в диапазоне [0,1]. */
    fun generate(
        context: Context,
        directory: File,
        prompt: String,
        steps: Int = 4,
        seed: Long,
        checkCancelled: () -> Unit = {},
        onStep: (Int, Int) -> Unit = { _, _ -> }
    ): FloatArray {
        require(steps in 4..5) { "Для этой LCM-модели доступны 4–5 шагов" }
        val tokenizer = ClipTokenizerFiles.load(File(directory, "tokenizer"))
        var unet: QnnUInt16Graph? = null
        try {
            val pipeline = LcmImagePipeline(
                textEncoder = { text -> encodeText(context, File(directory, "text_encoder.onnx"), tokenizer.tokenize(text)) },
                unet = { latent, timestep, embedding ->
                    val graph = unet ?: QnnUInt16Graph.open(context, File(directory, "unet.onnx"),
                        listOf(unetTime, unetLatent, unetText), unetOutput).also { unet = it }
                    graph.run(mapOf("timestep" to floatArrayOf(timestep.toFloat()), "latent" to latent, "text_emb" to embedding))
                },
                vae = { latent ->
                    unet?.close()
                    unet = null
                    checkCancelled()
                    QnnUInt16Graph.open(context, File(directory, "vae.onnx"), listOf(vaeInput), vaeOutput).use {
                        it.run(mapOf("latent" to latent))
                    }
                },
                decoderRange = LcmImagePipeline.DecoderRange.ZERO_TO_ONE
            )
            return pipeline.generate(prompt, steps, seed, checkCancelled, onStep)
        } finally {
            unet?.close()
        }
    }

    private fun encodeText(context: Context, file: File, tokens: IntArray): FloatArray {
        QnnHtpRuntime.open(context, file).use { htp ->
            val inputs = htp.session.inputInfo
            val outputs = htp.session.outputInfo
            require(inputs.keys == setOf("tokens")) { "Неподходящий граф CLIP" }
            val input = inputs.getValue("tokens").info as? TensorInfo ?: error("Вход CLIP не тензор")
            require(input.onnxType == TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
                input.shape.contentEquals(longArrayOf(1, 77))) { "Неверный тип или размер токенов CLIP" }
            val output = outputs[textOutput.name]?.info as? TensorInfo ?: error("Выход CLIP не найден")
            require(output.onnxType == TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 &&
                output.shape.contentEquals(textShape.toLongArray())) { "Неверный тип или размер выхода CLIP" }
            OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), IntBuffer.wrap(tokens), longArrayOf(1, 77)).use { tensor ->
                htp.session.run(mapOf("tokens" to tensor), setOf(textOutput.name)).use { result ->
                    val value = result.get(textOutput.name).orElseThrow { IllegalStateException("Выход CLIP отсутствует") } as? OnnxTensor
                        ?: error("Выход CLIP не тензор")
                    val data = value.shortBuffer ?: error("Выход CLIP нельзя прочитать")
                    require(data.remaining() == textOutput.elementCount)
                    return FloatArray(textOutput.elementCount) { textOutput.quantizer.dequantize(data.get().toInt() and 0xffff) }
                }
            }
        }
    }
}

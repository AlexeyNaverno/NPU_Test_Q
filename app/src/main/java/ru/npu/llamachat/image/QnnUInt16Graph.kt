package ru.npu.llamachat.image

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.UInt16TensorFactory
import android.content.Context
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Конфигурация каждого графа должна поступать из проверенного манифеста модели. */
internal data class QuantizedTensorSpec(
    val name: String,
    val shape: List<Long>,
    val quantizer: AffineUInt16Quantizer
) {
    init {
        require(name.isNotBlank() && shape.size <= TensorInfo.MAX_DIMENSIONS)
        var count = 1L
        shape.forEach {
            require(it > 0 && count <= Int.MAX_VALUE / 2L / it) { "Недопустимая форма тензора" }
            count *= it
        }
    }
    val elementCount: Int get() = shape.fold(1L) { a, b -> a * b }.toInt()
}

/**
 * Один строго HTP-граф с явно заданными именами, формами и параметрами quantization.
 * Не выбирает модель и не является готовым SD-пайплайном. CPU fallback в графе запрещён.
 */
internal class QnnUInt16Graph private constructor(
    private val htp: QnnHtpRuntime.HtpSession,
    private val inputs: List<QuantizedTensorSpec>,
    private val output: QuantizedTensorSpec
) : Closeable {
    companion object {
        fun open(context: Context, file: File, inputs: List<QuantizedTensorSpec>, output: QuantizedTensorSpec): QnnUInt16Graph {
            require(inputs.isNotEmpty() && inputs.map { it.name }.distinct().size == inputs.size)
            val session = QnnHtpRuntime.open(context, file)
            try {
                val actualInputs = session.session.inputInfo
                val actualOutputs = session.session.outputInfo
                require(actualInputs.keys == inputs.map { it.name }.toSet()) { "Входы графа не совпадают с манифестом" }
                require(actualOutputs.keys.contains(output.name)) { "Выход графа не найден" }
                inputs.forEach { spec -> validate(spec, actualInputs.getValue(spec.name).info) }
                validate(output, actualOutputs.getValue(output.name).info)
                return QnnUInt16Graph(session, inputs, output)
            } catch (t: Throwable) {
                session.close()
                throw t
            }
        }

        private fun validate(spec: QuantizedTensorSpec, info: ai.onnxruntime.ValueInfo) {
            require(info is TensorInfo && info.onnxType == TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
                "Тензор ${spec.name} не имеет тип UINT16"
            }
            require(info.shape.contentEquals(spec.shape.toLongArray())) { "Неверная форма тензора ${spec.name}" }
        }
    }

    /** Возвращает деквантованный выход; вызывающий отвечает за сериализацию и жизненный цикл. */
    fun run(values: Map<String, FloatArray>): FloatArray {
        require(values.keys == inputs.map { it.name }.toSet()) { "Неверный набор входов" }
        val tensors = LinkedHashMap<String, OnnxTensor>()
        try {
            for (spec in inputs) {
                val data = values.getValue(spec.name)
                require(data.size == spec.elementCount) { "Неверный размер входа ${spec.name}" }
                val buffer = ByteBuffer.allocateDirect(data.size * 2).order(ByteOrder.nativeOrder())
                data.forEach { buffer.putShort(spec.quantizer.quantize(it).toShort()) }
                buffer.flip()
                tensors[spec.name] = UInt16TensorFactory.create(OrtEnvironment.getEnvironment(), buffer, spec.shape.toLongArray())
            }
            htp.session.run(tensors, setOf(output.name)).use { result ->
                val tensor = result.get(output.name).orElseThrow { IllegalStateException("Выход графа отсутствует") } as? OnnxTensor
                    ?: error("Выход графа не является тензором")
                validate(output, tensor.info)
                val data = tensor.shortBuffer ?: error("Невозможно прочитать UINT16")
                require(data.remaining() == output.elementCount)
                return FloatArray(output.elementCount) { output.quantizer.dequantize(data.get().toInt() and 0xffff) }
            }
        } finally {
            tensors.values.forEach { it.close() }
        }
    }

    override fun close() = htp.close()
}

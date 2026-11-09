package ru.npu.llamachat.image

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.TensorInfo.OnnxTensorType
import ai.onnxruntime.UInt16TensorFactory
import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** CPU используется лишь для проверки JNI/формата UINT16 на хосте, не для диффузии. */
class UInt16HostInteropTest {
    @Test fun `ORT accepts UINT16 input and preserves high bits`() {
        val model = checkNotNull(javaClass.getResourceAsStream("/uint16_identity.onnx")).use { it.readBytes() }
        val env = OrtEnvironment.getEnvironment()
        val data = ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder())
        data.putShort(111).putShort(0).putShort(32768.toShort()).putShort(65535.toShort())
        data.position(2)
        env.createSession(model).use { session ->
            UInt16TensorFactory.create(env, data, longArrayOf(3)).use { input ->
                assertEquals(OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, input.info.onnxType)
                session.run(mapOf("x" to input)).use { result ->
                    val output = result[0] as OnnxTensor
                    assertEquals(OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, output.info.onnxType)
                    val values = output.shortBuffer!!
                    assertEquals(0, values.get().toInt() and 0xffff)
                    assertEquals(32768, values.get().toInt() and 0xffff)
                    assertEquals(65535, values.get().toInt() and 0xffff)
                }
            }
        }
    }
}

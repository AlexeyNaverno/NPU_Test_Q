package ru.npu.llamachat.image

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.TensorInfo.OnnxTensorType
import ai.onnxruntime.UInt16TensorFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.nio.ByteOrder

@RunWith(AndroidJUnit4::class)
class UInt16TensorFactoryTest {
    @Test fun createsNativeUint16AndPreservesBits() {
        val env = OrtEnvironment.getEnvironment()
        val data = ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder())
        data.putShort(123.toShort()).putShort(0.toShort()).putShort(32768.toShort()).putShort(65535.toShort())
        data.position(2)
        UInt16TensorFactory.create(env, data, longArrayOf(3)).use { tensor ->
            assertEquals(OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, tensor.info.onnxType)
            val values = tensor.shortBuffer!!
            assertEquals(0, values.get().toInt() and 0xffff)
            assertEquals(32768, values.get().toInt() and 0xffff)
            assertEquals(65535, values.get().toInt() and 0xffff)
        }
    }

    @Test fun runsUint16IdentityOnOrtCpuForInteropOnly() {
        // CPU только для проверки передачи типа; это не тест QNN/NPU и не генерация изображений.
        val env = OrtEnvironment.getEnvironment()
        val model = InstrumentationRegistry.getInstrumentation().context.assets.open("uint16_identity.onnx").use { it.readBytes() }
        val data = ByteBuffer.allocateDirect(6).order(ByteOrder.nativeOrder())
        data.putShort(0).putShort(32768.toShort()).putShort(65535.toShort()).flip()
        env.createSession(model).use { session ->
            UInt16TensorFactory.create(env, data, longArrayOf(3)).use { input ->
                session.run(mapOf("x" to input)).use { result ->
                    val output = result[0] as ai.onnxruntime.OnnxTensor
                    assertEquals(OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, output.info.onnxType)
                    val values = output.shortBuffer!!
                    assertEquals(0, values.get().toInt() and 0xffff)
                    assertEquals(32768, values.get().toInt() and 0xffff)
                    assertEquals(65535, values.get().toInt() and 0xffff)
                }
            }
        }
    }

    @Test fun rejectsInvalidShapeAndHeapBuffer() {
        val env = OrtEnvironment.getEnvironment()
        val direct = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder())
        assertThrows(IllegalArgumentException::class.java) { UInt16TensorFactory.create(env, direct, longArrayOf(3)) }
        assertThrows(IllegalArgumentException::class.java) { UInt16TensorFactory.create(env, direct, longArrayOf(Long.MAX_VALUE, 2)) }
        assertThrows(IllegalArgumentException::class.java) { UInt16TensorFactory.create(env, ByteBuffer.allocate(4), longArrayOf(2)) }
    }
}

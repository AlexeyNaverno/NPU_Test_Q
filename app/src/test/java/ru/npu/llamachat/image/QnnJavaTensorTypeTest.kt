package ru.npu.llamachat.image

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.TensorInfo.OnnxTensorType
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class QnnJavaTensorTypeTest {
    @Test fun `ORT Java 1_29 maps UINT16 output to signed Java carrier`() {
        assertEquals(OnnxJavaType.INT16, OnnxJavaType.mapFromOnnxTensorType(OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16))
        assertFalse(OnnxJavaType.values().any { it.name == "UINT16" })
    }
}

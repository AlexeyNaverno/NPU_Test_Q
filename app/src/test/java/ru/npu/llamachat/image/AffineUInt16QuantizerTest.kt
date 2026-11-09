package ru.npu.llamachat.image

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class AffineUInt16QuantizerTest {
    @Test fun `quantize uses zero point and round to nearest even`() {
        val q = AffineUInt16Quantizer(0.5, 32768)
        assertEquals(32768, q.quantize(0f))
        assertEquals(32768, q.quantize(0.25f))
        assertEquals(32770, q.quantize(0.75f))
        assertEquals(32766, q.quantize(-0.75f))
        assertEquals(-1f, q.dequantize(32766), 0f)
    }

    @Test fun `values saturate at unsigned limits`() {
        val q = AffineUInt16Quantizer(0.001, 30000)
        assertEquals(0, q.quantize(-1e20f))
        assertEquals(65535, q.quantize(1e20f))
    }

    @Test fun `invalid parameters and non finite tensors are rejected`() {
        assertThrows(IllegalArgumentException::class.java) { AffineUInt16Quantizer(0.0, 0) }
        assertThrows(IllegalArgumentException::class.java) { AffineUInt16Quantizer(Double.NaN, 0) }
        assertThrows(IllegalArgumentException::class.java) { AffineUInt16Quantizer(1.0, 65536) }
        val q = AffineUInt16Quantizer(1.0, 0)
        assertThrows(IllegalArgumentException::class.java) { q.quantize(Float.NaN) }
        assertThrows(IllegalArgumentException::class.java) { q.dequantize(-1) }
    }
}

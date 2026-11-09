package ru.npu.llamachat.image

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class QuantizedTensorSpecTest {
    private val q = AffineUInt16Quantizer(0.5, 32768)

    @Test fun `valid shape computes element count including scalar`() {
        assertEquals(24, QuantizedTensorSpec("latent", listOf(1, 2, 3, 4), q).elementCount)
        assertEquals(1, QuantizedTensorSpec("time", emptyList(), q).elementCount)
    }

    @Test fun `invalid and too large shapes fail before allocation`() {
        assertThrows(IllegalArgumentException::class.java) { QuantizedTensorSpec("", listOf(1), q) }
        assertThrows(IllegalArgumentException::class.java) { QuantizedTensorSpec("x", listOf(0), q) }
        assertThrows(IllegalArgumentException::class.java) { QuantizedTensorSpec("x", listOf(-1), q) }
        assertThrows(IllegalArgumentException::class.java) { QuantizedTensorSpec("x", listOf(Long.MAX_VALUE, 2), q) }
        assertThrows(IllegalArgumentException::class.java) { QuantizedTensorSpec("x", List(9) { 1 }, q) }
    }
}

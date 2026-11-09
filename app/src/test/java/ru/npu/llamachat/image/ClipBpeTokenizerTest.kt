package ru.npu.llamachat.image

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class ClipBpeTokenizerTest {
    private val vocab = mapOf(
        "<|startoftext|>" to 49406, "<|endoftext|>" to 49407,
        "hi</w>" to 10, "!</w>" to 11, "1</w>" to 12, "2</w>" to 13,
        "a</w>" to 14, "Ð" to 15, "º</w>" to 16
    )
    private val tokenizer = ClipBpeTokenizer(vocab, listOf("h" to "i</w>"))

    @Test fun `BPE lowercases pads and keeps digits separate`() {
        val result = tokenizer.tokenize("HI! 12")
        assertEquals(77, result.size)
        assertEquals(listOf(49406, 10, 11, 12, 13, 49407), result.take(6))
        assertEquals(49407, result.last())
    }

    @Test fun `UTF8 uses CLIP byte to unicode mapping`() {
        assertEquals(listOf(49406, 15, 16, 49407), tokenizer.tokenize("к").take(4))
    }

    @Test fun `truncation reserves EOS and rejects oversized inputs`() {
        val result = tokenizer.tokenize(List(100) { "a" }.joinToString(" "))
        assertEquals(49406, result.first())
        assertEquals(49407, result.last())
        assertEquals(75, result.count { it == 14 })
        assertThrows(IllegalArgumentException::class.java) { tokenizer.tokenize("a".repeat(4097)) }
    }

    @Test fun `rejects bad model data and unknown tokens`() {
        assertThrows(IllegalArgumentException::class.java) { ClipBpeTokenizer(emptyMap(), emptyList()) }
        assertThrows(IllegalArgumentException::class.java) { ClipBpeTokenizer(vocab, listOf("h" to "i", "h" to "i")) }
        assertThrows(IllegalArgumentException::class.java) { tokenizer.tokenize("z") }
    }
}

package ru.npu.llamachat.image

import org.junit.Assert.assertEquals
import org.junit.Test

/** Малые векторы и необходимые BPE-слияния из OpenAI CLIP simple_tokenizer, ревизия d05afc4. */
class ClipBpeReferenceTest {
    @Test fun `matches OpenAI CLIP for ordinary text and Cyrillic`() {
        val vocabulary = mapOf(
            "!</w>" to 256,
            ",</w>" to 267,
            "1</w>" to 272,
            "2</w>" to 273,
            "3</w>" to 274,
            "4</w>" to 275,
            "5</w>" to 276,
            "<|endoftext|>" to 49407,
            "<|startoftext|>" to 49406,
            "a</w>" to 320,
            "anime</w>" to 6469,
            "blue</w>" to 1746,
            "cat</w>" to 2368,
            "girl</w>" to 1611,
            "hair</w>" to 2225,
            "of</w>" to 539,
            "photo</w>" to 1125,
            "with</w>" to 593,
            "²" to 110,
            "Ðµ" to 22176,
            "Ð¸" to 16701,
            "Ð¸Ð" to 29503,
            "Ð¼" to 38018,
            "Ð¿" to 26302,
            "Ñ" to 141,
            "ÑĢ" to 16370,
            "Ģ</w>" to 478,
            "Ĥ</w>" to 480
        )
        val merges = listOf(
            "a" to "n",
            "i" to "t",
            "t" to "o</w>",
            "a" to "t</w>",
            "o" to "f</w>",
            "h" to "a",
            "i" to "r",
            "w" to "it",
            "wit" to "h</w>",
            "h" to "o",
            "m" to "e</w>",
            "l" to "u",
            "i" to "r</w>",
            "p" to "ho",
            "g" to "ir",
            "pho" to "to</w>",
            "b" to "lu",
            "an" to "i",
            "gir" to "l</w>",
            "blu" to "e</w>",
            "ha" to "ir</w>",
            "c" to "at</w>",
            "ani" to "me</w>",
            "Ñ" to "Ģ",
            "Ð" to "¸",
            "Ð" to "µ",
            "Ð" to "¿",
            "Ð¸" to "Ð",
            "Ð" to "¼"
        )
        val tokenizer = ClipBpeTokenizer(vocabulary, merges)
        assertEquals(listOf(320, 1125, 539, 320, 2368), tokenizer.tokenize("a photo of a cat").drop(1).take(5))
        assertEquals(listOf(26302, 16370, 29503, 110, 22176, 141, 480, 267, 38018, 16701, 141, 478, 256), tokenizer.tokenize("Привет, мир!").drop(1).take(13))
        assertEquals(listOf(6469, 1611, 593, 1746, 2225), tokenizer.tokenize("anime girl with blue hair").drop(1).take(5))
        assertEquals(listOf(272, 273, 274, 275, 276), tokenizer.tokenize("12345").drop(1).take(5))
    }
}

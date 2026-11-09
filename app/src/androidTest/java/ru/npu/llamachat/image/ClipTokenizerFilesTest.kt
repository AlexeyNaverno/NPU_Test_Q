package ru.npu.llamachat.image

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class ClipTokenizerFilesTest {
    @Test fun loadsTokenizerAndRejectsMalformedData() {
        val directory = File(InstrumentationRegistry.getInstrumentation().targetContext.cacheDir, "clip-test-${System.nanoTime()}")
        check(directory.mkdir())
        try {
            File(directory, "vocab.json").writeText("""{"<|startoftext|>":49406,"<|endoftext|>":49407,"hi</w>":10}""")
            File(directory, "merges.txt").writeText("#version: 0.2\nh i</w>\n")
            assertEquals(listOf(49406, 10, 49407), ClipTokenizerFiles.load(directory).tokenize("hi").take(3))
            File(directory, "merges.txt").writeText("broken\n")
            assertThrows(IllegalArgumentException::class.java) { ClipTokenizerFiles.load(directory) }
        } finally {
            directory.deleteRecursively()
        }
    }
}

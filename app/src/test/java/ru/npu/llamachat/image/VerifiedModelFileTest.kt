package ru.npu.llamachat.image

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.security.MessageDigest
import java.util.concurrent.CancellationException

class VerifiedModelFileTest {
    private val bytes = "abc".toByteArray()
    private fun hash(bytes: ByteArray, algorithm: String) = MessageDigest.getInstance(algorithm).digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 255) }

    @Test fun `checks SHA256 and reports progress`() {
        val out = ByteArrayOutputStream()
        var count = 0L
        VerifiedModelFile.copy(ByteArrayInputStream(bytes), out, ModelFile("test", 3, hash(bytes, "SHA-256")), {}, { count += it })
        assertEquals(3L, count)
        assertArrayEquals(bytes, out.toByteArray())
    }

    @Test fun `checks git blob and rejects corrupt or oversized content`() {
        val digest = hash("blob 3\u0000abc".toByteArray(), "SHA-1")
        val entry = ModelFile("test", 3, digest, true)
        VerifiedModelFile.copy(ByteArrayInputStream(bytes), ByteArrayOutputStream(), entry, {}, {})
        assertThrows(IllegalArgumentException::class.java) {
            VerifiedModelFile.copy(ByteArrayInputStream("abd".toByteArray()), ByteArrayOutputStream(), entry, {}, {})
        }
        assertThrows(IllegalArgumentException::class.java) {
            VerifiedModelFile.copy(ByteArrayInputStream("abcd".toByteArray()), ByteArrayOutputStream(), entry, {}, {})
        }
        assertThrows(IllegalArgumentException::class.java) {
            VerifiedModelFile.copy(ByteArrayInputStream("ab".toByteArray()), ByteArrayOutputStream(), entry, {}, {})
        }
    }

    @Test fun `cancellation and manifests`() {
        assertThrows(CancellationException::class.java) {
            VerifiedModelFile.copy(ByteArrayInputStream(bytes), ByteArrayOutputStream(), ModelFile("x", 3, "bad"), { throw CancellationException() }, {})
        }
        ImageModelCatalog.models.forEach { model ->
            assertEquals(model.files.size, model.files.map { it.path }.distinct().size)
            assertEquals(true, model.totalBytes > 1_000_000_000L)
            model.files.forEach { file ->
                assertEquals(if (file.gitBlob) 40 else 64, file.digest.length)
                assertEquals(false, file.path.startsWith('/') || file.path.split('/').contains(".."))
            }
        }
    }
}

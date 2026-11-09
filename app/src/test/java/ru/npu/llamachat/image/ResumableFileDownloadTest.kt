package ru.npu.llamachat.image

import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayInputStream
import java.io.IOException
import java.io.InputStream
import java.security.MessageDigest
import java.util.concurrent.CancellationException

class ResumableFileDownloadTest {
    @get:Rule val tmp = TemporaryFolder()
    private val bytes = "abcdefghij".toByteArray()
    private val digest = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it.toInt() and 255) }
    private val entry = ModelFile("test", bytes.size.toLong(), digest)
    private fun response(code: Int, range: String?, stream: InputStream) = object : ResumableFileDownload.Response {
        override val status = code
        override val range = range
        override val encoding: String? = null
        override fun stream() = stream
        override fun close() = stream.close()
    }
    private fun ranged(start: Long, end: Long): ResumableFileDownload.Response {
        val data = bytes.copyOfRange(start.toInt(), end.toInt() + 1)
        return response(206, "bytes $start-$end/${bytes.size}", ByteArrayInputStream(data))
    }
    private val noWait: (Long) -> Unit = {}

    @Test fun `resumes existing file and verifies checksum`() {
        val file = tmp.newFile()
        file.writeBytes(bytes.copyOf(4))
        val starts = mutableListOf<Long>()
        ResumableFileDownload.download(entry, file, { start, end -> starts.add(start); ranged(start, end) }, {}, {}, noWait)
        assertEquals(listOf(4L), starts)
        assertArrayEquals(bytes, file.readBytes())
    }

    @Test fun `retries after connection abort with preserved bytes`() {
        val file = tmp.newFile()
        val starts = mutableListOf<Long>()
        ResumableFileDownload.download(entry, file, { start, end ->
            starts.add(start)
            if (starts.size == 1) response(206, "bytes 0-9/10", object : InputStream() {
                var index = 0
                override fun read(): Int = if (index < 4) bytes[index++].toInt() else throw IOException("connection abort")
                override fun read(b: ByteArray, off: Int, len: Int): Int {
                    if (index >= 4) throw IOException("connection abort")
                    val n = minOf(len, 4 - index)
                    bytes.copyInto(b, off, index, index + n); index += n; return n
                }
            }) else ranged(start, end)
        }, {}, {}, noWait)
        assertEquals(listOf(0L, 4L), starts)
        assertArrayEquals(bytes, file.readBytes())
    }

    @Test fun `large files are requested in bounded chunks`() {
        val data = ByteArray(8 * 1024 * 1024 + 23) { (it % 251).toByte() }
        val hash = MessageDigest.getInstance("SHA-256").digest(data).joinToString("") { "%02x".format(it.toInt() and 255) }
        val large = ModelFile("large", data.size.toLong(), hash)
        val ranges = mutableListOf<Pair<Long, Long>>()
        val file = tmp.newFile()
        ResumableFileDownload.download(large, file, { start, end ->
            ranges.add(start to end)
            response(206, "bytes $start-$end/${data.size}", ByteArrayInputStream(data.copyOfRange(start.toInt(), end.toInt() + 1)))
        }, {}, {}, noWait)
        assertEquals(listOf(0L to (8L * 1024 * 1024 - 1), (8L * 1024 * 1024) to (data.size - 1L)), ranges)
        assertArrayEquals(data, file.readBytes())
    }

    @Test fun `server ignoring range restarts without appending duplicate data`() {
        val file = tmp.newFile().apply { writeBytes(bytes.copyOf(3)) }
        ResumableFileDownload.download(entry, file, { _, _ -> response(200, null, ByteArrayInputStream(bytes)) }, {}, {}, noWait)
        assertArrayEquals(bytes, file.readBytes())
    }

    @Test fun `wrong content range retried but never appended`() {
        val file = tmp.newFile().apply { writeBytes(bytes.copyOf(3)) }
        var calls = 0
        assertThrows(IOException::class.java) {
            ResumableFileDownload.download(entry, file, { _, _ -> calls++; response(206, "bytes 0-9/10", ByteArrayInputStream(bytes)) }, {}, {}, noWait)
        }
        assertEquals(5, calls)
        assertArrayEquals(bytes.copyOf(3), file.readBytes())
    }

    @Test fun `corrupt completed file is replaced and cancellation keeps partial`() {
        val file = tmp.newFile().apply { writeBytes("xxxxxxxxxx".toByteArray()) }
        ResumableFileDownload.download(entry, file, { start, end -> ranged(start, end) }, {}, {}, noWait)
        assertArrayEquals(bytes, file.readBytes())
        file.writeBytes(bytes.copyOf(4))
        assertThrows(CancellationException::class.java) {
            ResumableFileDownload.download(entry, file, { _, _ -> throw AssertionError("network not expected") }, { throw CancellationException() }, {}, noWait)
        }
        assertArrayEquals(bytes.copyOf(4), file.readBytes())
    }
}

package ru.npu.llamachat.image

import java.io.Closeable
import java.io.EOFException
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream

/** Скачивает один файл диапазонами, сохраняя частичные данные до проверки целого файла. */
internal object ResumableFileDownload {
    private const val CHUNK = 8L * 1024 * 1024
    private const val MAX_FAILURES = 5
    private val contentRange = Regex("bytes (\\d+)-(\\d+)/(\\d+)")

    interface Response : Closeable {
        val status: Int
        val range: String?
        val encoding: String?
        fun stream(): InputStream
    }
    fun interface Transport { fun open(start: Long, end: Long): Response }

    fun download(entry: ModelFile, file: File, transport: Transport,
                 checkCancelled: () -> Unit, progress: (Long) -> Unit,
                 wait: (Long) -> Unit = { Thread.sleep(it) },
                 onRetry: (Int, String) -> Unit = { _, _ -> },
                 onVerify: () -> Unit = {}) {
        require(entry.bytes > 0)
        if (file.exists() && file.length() > entry.bytes) check(file.delete()) { "Не удалось удалить повреждённый файл" }
        var checksumFailures = 0
        var failures = 0
        while (true) {
            checkCancelled()
            val start = file.length()
            if (start == entry.bytes) {
                onVerify()
                try {
                    file.inputStream().buffered().use { input ->
                        VerifiedModelFile.copy(input, object : OutputStream() {
                            override fun write(b: Int) = Unit
                            override fun write(b: ByteArray, off: Int, len: Int) = Unit
                        }, entry, checkCancelled) { }
                    }
                    progress(entry.bytes)
                    return
                } catch (e: IllegalArgumentException) {
                    check(file.delete()) { "Не удалось удалить повреждённый файл ${entry.path}" }
                    progress(0)
                    if (++checksumFailures > 1) throw IOException("Контрольная сумма ${entry.path} не совпала после повторной загрузки", e)
                    continue
                }
            }
            val end = minOf(start + CHUNK - 1, entry.bytes - 1)
            try {
                transport.open(start, end).use { response ->
                    if (response.encoding != null && !response.encoding.equals("identity", true)) throw IOException("Сервер вернул сжатый файл")
                    val expected: Long
                    val append: Boolean
                    when (response.status) {
                        206 -> {
                            val m = contentRange.matchEntire(response.range.orEmpty()) ?: throw IOException("Некорректный Content-Range")
                            val from = m.groupValues[1].toLongOrNull()
                            val to = m.groupValues[2].toLongOrNull()
                            val total = m.groupValues[3].toLongOrNull()
                            if (from != start || to == null || to < start || to > end || total != entry.bytes) throw IOException("Сервер вернул неверный диапазон")
                            expected = to - start + 1
                            append = start > 0
                        }
                        200 -> {
                            // Сервер проигнорировал Range: нельзя дописывать ответ к старым байтам.
                            expected = entry.bytes
                            append = false
                            if (start > 0) progress(0)
                        }
                        else -> throw IOException("Сервер вернул HTTP ${response.status}")
                    }
                    response.stream().use { input ->
                        file.outputStreamAppend(append).buffered().use { output ->
                            val buffer = ByteArray(256 * 1024)
                            var received = 0L
                            while (received < expected) {
                                checkCancelled()
                                val n = input.read(buffer, 0, minOf(buffer.size.toLong(), expected - received).toInt())
                                if (n < 0) throw EOFException("Соединение прервалось при загрузке ${entry.path}")
                                if (n == 0) continue
                                output.write(buffer, 0, n)
                                received += n
                                progress((if (append) start else 0) + received)
                            }
                            if (input.read() != -1) throw IOException("Сервер вернул лишние данные")
                        }
                    }
                }
                failures = 0
            } catch (e: IOException) {
                checkCancelled()
                if (++failures >= MAX_FAILURES) throw IOException("Не удалось скачать ${entry.path} после $MAX_FAILURES попыток: ${e.message}", e)
                onRetry(failures, e.message.orEmpty())
                var remaining = (1L shl (failures - 1)) * 1000L
                while (remaining > 0) {
                    checkCancelled()
                    val interval = minOf(remaining, 250L)
                    wait(interval)
                    remaining -= interval
                }
            }
        }
    }

    private fun File.outputStreamAppend(append: Boolean) = java.io.FileOutputStream(this, append)
}

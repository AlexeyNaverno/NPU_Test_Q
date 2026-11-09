package ru.npu.llamachat.image

import java.io.InputStream
import java.io.OutputStream
import java.security.MessageDigest

/** Потоковая проверка размера и контрольной суммы до установки файла. */
internal object VerifiedModelFile {
    fun copy(input: InputStream, output: OutputStream, entry: ModelFile, checkCancelled: () -> Unit, progress: (Long) -> Unit) {
        require(entry.bytes > 0 && entry.path.isNotBlank())
        val digest = MessageDigest.getInstance(if (entry.gitBlob) "SHA-1" else "SHA-256")
        if (entry.gitBlob) digest.update("blob ${entry.bytes}\u0000".toByteArray(Charsets.UTF_8))
        val buffer = ByteArray(256 * 1024)
        var count = 0L
        while (true) {
            checkCancelled()
            val n = input.read(buffer)
            if (n < 0) break
            if (n == 0) continue
            count += n
            require(count <= entry.bytes) { "Неверный размер ${entry.path}" }
            digest.update(buffer, 0, n)
            output.write(buffer, 0, n)
            progress(n.toLong())
        }
        val actual = digest.digest().joinToString("") { "%02x".format(it.toInt() and 255) }
        require(count == entry.bytes && actual == entry.digest) { "Размер или контрольная сумма ${entry.path} не совпадают" }
    }
}

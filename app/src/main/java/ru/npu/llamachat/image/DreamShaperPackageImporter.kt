package ru.npu.llamachat.image

import android.content.Context
import android.net.Uri
import android.provider.DocumentsContract
import java.io.File
import java.security.MessageDigest
import java.util.UUID

/** Импорт только закреплённой ревизии модели через SAF, без доверия именам или размерам из провайдера. */
internal object DreamShaperPackageImporter {
    private data class Entry(val path: String, val bytes: Long, val digest: String, val gitBlob: Boolean = false)
    private val entries = listOf(
        Entry("text_encoder.onnx", 733, "fc06df873065d0129fcb1e42b85200486bc9414b6914f2f82ea69a8c8ed6c8b8"),
        Entry("text_encoder_qairt_context.bin", 178077696, "b079a8d6dbd8613010e1027bcc5c5dfa73e09e4e67dcf6aeb3c1f2b004cd9cca"),
        Entry("unet.onnx", 1714, "286a49715a74e7fc739ae972f27965b70458d62130b3f83e1c28d65eec30121b"),
        Entry("unet_qairt_context.bin", 882466816, "86285ff4e4cda4fbe69ea9d2997b308c5de4a960234b28d56ca5a4231500a316"),
        Entry("vae.onnx", 873, "a92718467028e6289be68e46e7000f5e99d0d14377ebfbec738cee8a05bef29d"),
        Entry("vae_qairt_context.bin", 67473408, "4aba13be2a85e892053cd9b52a3f5a12415f30d6ae9ec1e438ce593f2131e531"),
        Entry("tokenizer/vocab.json", 1059962, "469be27c5c010538f845f518c4f5e8574c78f7c8", true),
        Entry("tokenizer/merges.txt", 524619, "76e821f1b6f0a9709293c3b6b51ed90980b3166b", true)
    )
    private const val FOLDER = "dreamshaper8-lcm-v73"
    fun installedDirectory(context: Context) = File(context.filesDir, FOLDER)

    /** Дорогостоящая операция; вызывается в IO-потоке и только с эксклюзивной блокировкой генерации. */
    fun import(context: Context, tree: Uri, checkCancelled: () -> Unit, progress: (Long, Long) -> Unit): File {
        val resolver = context.contentResolver
        val rootId = DocumentsContract.getTreeDocumentId(tree)
        val stage = File(context.filesDir, ".$FOLDER-${UUID.randomUUID()}")
        check(stage.mkdir()) { "Не удалось создать временную папку модели" }
        val total = entries.sumOf { it.bytes }
        var copied = 0L
        try {
            for (entry in entries) {
                checkCancelled()
                var parent = rootId
                val parts = entry.path.split('/')
                for (part in parts.dropLast(1)) parent = findChild(context, tree, parent, part, directory = true)
                val child = findChild(context, tree, parent, parts.last(), directory = false)
                val uri = DocumentsContract.buildDocumentUriUsingTree(tree, child)
                val dest = File(stage, entry.path)
                dest.parentFile?.mkdirs()
                val digest = MessageDigest.getInstance(if (entry.gitBlob) "SHA-1" else "SHA-256")
                if (entry.gitBlob) digest.update("blob ${entry.bytes}\u0000".toByteArray(Charsets.UTF_8))
                var count = 0L
                val input = resolver.openInputStream(uri) ?: error("Не удалось открыть ${entry.path}")
                input.use { src ->
                    dest.outputStream().buffered().use { out ->
                        val buffer = ByteArray(256 * 1024)
                        while (true) {
                            checkCancelled()
                            val n = src.read(buffer)
                            if (n < 0) break
                            if (n == 0) continue
                            count += n
                            require(count <= entry.bytes) { "Неверный размер ${entry.path}" }
                            digest.update(buffer, 0, n)
                            out.write(buffer, 0, n)
                            copied += n
                            progress(copied, total)
                        }
                    }
                }
                val actual = digest.digest().joinToString("") { "%02x".format(it.toInt() and 255) }
                require(count == entry.bytes && actual == entry.digest) { "Размер или контрольная сумма ${entry.path} не совпадают с моделью" }
            }
            checkCancelled()
            val target = installedDirectory(context)
            val backup = File(context.filesDir, ".$FOLDER-old-${UUID.randomUUID()}")
            if (target.exists()) check(target.renameTo(backup)) { "Не удалось заменить старую модель" }
            if (!stage.renameTo(target)) {
                if (backup.exists()) check(backup.renameTo(target)) { "Не удалось восстановить старую модель" }
                error("Не удалось сохранить модель")
            }
            backup.deleteRecursively()
            return target
        } finally {
            stage.deleteRecursively()
        }
    }

    private fun findChild(context: Context, tree: Uri, parent: String, name: String, directory: Boolean): String {
        val uri = DocumentsContract.buildChildDocumentsUriUsingTree(tree, parent)
        val columns = arrayOf(DocumentsContract.Document.COLUMN_DOCUMENT_ID, DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_MIME_TYPE)
        var match: String? = null
        val cursor = context.contentResolver.query(uri, columns, null, null, null) ?: error("Невозможно прочитать папку модели")
        cursor.use {
            while (it.moveToNext()) {
                if (it.getString(1) == name) {
                    val isDirectory = it.getString(2) == DocumentsContract.Document.MIME_TYPE_DIR
                    require(isDirectory == directory && match == null) { "Неоднозначный или неверный элемент $name" }
                    match = it.getString(0)
                }
            }
        }
        return match ?: error("В папке модели отсутствует $name")
    }
}

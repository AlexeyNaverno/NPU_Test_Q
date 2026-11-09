package ru.npu.llamachat.image

import android.content.Context
import android.net.Uri
import android.provider.DocumentsContract
import java.io.File
import java.io.OutputStream
import java.util.UUID

/** Сохраняет проверенную копию в новой папке SAF, не изменяя существующие пользовательские файлы. */
internal object ExternalModelStorage {
    fun export(context: Context, tree: Uri, model: ImageModel, source: File,
               checkCancelled: () -> Unit, onFile: (String) -> Unit,
               progress: (Long, Long) -> Unit): String {
        require(model in ImageModelCatalog.models)
        val resolver = context.contentResolver
        val treeRoot = DocumentsContract.buildDocumentUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree))
        val name = "NPU-${model.id}-${UUID.randomUUID().toString().take(8)}"
        val root = DocumentsContract.createDocument(resolver, treeRoot, DocumentsContract.Document.MIME_TYPE_DIR, name)
            ?: error("Не удалось создать папку $name в выбранном месте")
        val dirs = mutableMapOf("" to root)
        var completed = 0L
        try {
            for (entry in model.files) {
                checkCancelled()
                onFile(entry.path)
                val parts = entry.path.split('/')
                var path = ""
                var parent = root
                for (part in parts.dropLast(1)) {
                    path = if (path.isEmpty()) part else "$path/$part"
                    parent = dirs[path] ?: (DocumentsContract.createDocument(resolver, parent,
                        DocumentsContract.Document.MIME_TYPE_DIR, part)
                        ?: error("Не удалось создать папку $path")).also { dirs[path] = it }
                }
                val destination = DocumentsContract.createDocument(resolver, parent, "application/octet-stream", parts.last())
                    ?: error("Не удалось создать ${entry.path}")
                val src = File(source, entry.path)
                require(src.isFile && src.length() == entry.bytes) { "Отсутствует локальный файл ${entry.path}" }
                val base = completed
                val out = resolver.openOutputStream(destination, "w") ?: error("Не удалось записать ${entry.path}")
                src.inputStream().buffered().use { input ->
                    out.buffered().use { output ->
                        VerifiedModelFile.copy(input, output, entry, checkCancelled) { n ->
                            completed += n
                            progress(completed, model.totalBytes)
                        }
                    }
                }
                check(completed == base + entry.bytes)
                checkCancelled()
                onFile("проверка ${entry.path}")
                val copied = resolver.openInputStream(destination) ?: error("Не удалось проверить ${entry.path}")
                copied.buffered().use { input ->
                    VerifiedModelFile.copy(input, object : OutputStream() {
                        override fun write(b: Int) = Unit
                        override fun write(b: ByteArray, off: Int, len: Int) = Unit
                    }, entry, checkCancelled) { }
                }
            }
            return name
        } catch (e: Exception) {
            if (e is java.util.concurrent.CancellationException) throw e
            throw IllegalStateException("Не удалось полностью сохранить модель в папке $name: ${e.message}. Неполную папку можно удалить вручную.", e)
        }
    }
}

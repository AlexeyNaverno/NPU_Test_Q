package ru.npu.llamachat.image

import android.content.Context
import android.net.Uri
import android.provider.DocumentsContract
import java.io.File
import java.util.UUID

/** Импорт закреплённого комплекта через SAF с проверкой каждого файла. */
internal object DreamShaperPackageImporter {
    fun installedDirectory(context: Context) = ImageModelDownloader.directory(context, ImageModelCatalog.dreamShaper)

    /** Дорогостоящая операция; вызывается в IO-потоке и только с эксклюзивной блокировкой генерации. */
    fun import(context: Context, tree: Uri, model: ImageModel, checkCancelled: () -> Unit, progress: (Long, Long) -> Unit): File {
        require(model in ImageModelCatalog.models)
        ImageModelDownloader.recoverBackup(context, model)
        val entries = model.files
        val folder = model.folder
        require(context.filesDir.usableSpace >= model.totalBytes + 64L * 1024 * 1024) {
            "Для внутренней рабочей копии нужно ещё около ${(model.totalBytes / 1024 / 1024) + 64} МиБ"
        }
        val resolver = context.contentResolver
        val rootId = DocumentsContract.getTreeDocumentId(tree)
        val stage = File(context.filesDir, ".$folder-${UUID.randomUUID()}")
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
                val input = resolver.openInputStream(uri) ?: error("Не удалось открыть ${entry.path}")
                input.use { src ->
                    dest.outputStream().buffered().use { out ->
                        VerifiedModelFile.copy(src, out, entry, checkCancelled) { n ->
                            copied += n
                            progress(copied, total)
                        }
                    }
                }
            }
            checkCancelled()
            val target = ImageModelDownloader.directory(context, model)
            val backup = File(context.filesDir, ".$folder-old-${UUID.randomUUID()}")
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

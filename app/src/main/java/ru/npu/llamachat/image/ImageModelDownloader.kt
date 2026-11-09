package ru.npu.llamachat.image

import android.content.Context
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID
import javax.net.ssl.HttpsURLConnection

/** Загрузка закреплённой ревизии во внутреннюю память, без публикации неполной модели. */
internal object ImageModelDownloader {
    fun directory(context: Context, model: ImageModel): File = File(context.filesDir, model.folder)

    fun isInstalled(context: Context, model: ImageModel): Boolean {
        val root = directory(context, model)
        return root.isDirectory && model.files.all { File(root, it.path).let { f -> f.isFile && f.length() == it.bytes } }
    }

    /** После прерывания между переименованиями восстанавливаем единственный старый пакет. */
    fun recoverBackup(context: Context, model: ImageModel) = recoverBackupIn(context.filesDir, model)

    fun recoverBackupIn(root: File, model: ImageModel) {
        val target = File(root, model.folder)
        if (!target.exists()) {
            val backups = root.listFiles()?.filter { it.isDirectory && it.name.startsWith(".${model.folder}-old-") }.orEmpty()
            require(backups.size <= 1) { "Найдено несколько резервных копий модели; восстановление требует ручной проверки" }
            if (backups.size == 1) check(backups[0].renameTo(target)) { "Не удалось восстановить предыдущую модель" }
        }
    }

    fun download(context: Context, model: ImageModel, checkCancelled: () -> Unit, progress: (Long, Long) -> Unit): File {
        require(model in ImageModelCatalog.models)
        recoverBackup(context, model)
        require(context.filesDir.usableSpace >= model.totalBytes + 64L * 1024 * 1024) {
            "Недостаточно свободного места: необходимо ещё около ${(model.totalBytes / 1024 / 1024) + 64} МиБ"
        }
        // Удаляются только незавершённые загрузки этой модели, не установленная модель или бэкапы.
        context.filesDir.listFiles()?.filter { it.name.startsWith(".${model.folder}-download-") }?.forEach { it.deleteRecursively() }
        val stage = File(context.filesDir, ".${model.folder}-download-${UUID.randomUUID()}")
        check(stage.mkdir()) { "Не удалось создать временную папку" }
        var copied = 0L
        try {
            for (entry in model.files) {
                checkCancelled()
                val dest = File(stage, entry.path)
                check(dest.canonicalPath.startsWith(stage.canonicalPath + File.separator)) { "Неверный путь файла модели" }
                check(dest.parentFile?.let { it.isDirectory || it.mkdirs() } == true) { "Не удалось создать папку" }
                val connection = connect("https://huggingface.co/${model.repository}/resolve/${model.revision}/${entry.path}")
                try {
                    connection.inputStream.use { input ->
                        dest.outputStream().buffered().use { out ->
                            VerifiedModelFile.copy(input, out, entry, checkCancelled) { n ->
                                copied += n
                                progress(copied, model.totalBytes)
                            }
                        }
                    }
                } finally { connection.disconnect() }
            }
            checkCancelled()
            val target = directory(context, model)
            val backup = File(context.filesDir, ".${model.folder}-old-${UUID.randomUUID()}")
            if (target.exists()) check(target.renameTo(backup)) { "Не удалось временно сохранить старую модель" }
            if (!stage.renameTo(target)) {
                if (backup.exists()) check(backup.renameTo(target)) { "Не удалось восстановить старую модель: ${backup.name}" }
                error("Не удалось установить модель")
            }
            backup.deleteRecursively()
            return target
        } finally { stage.deleteRecursively() }
    }

    private fun connect(initialUrl: String): HttpsURLConnection {
        var url = URL(initialUrl)
        repeat(6) {
            require(url.protocol == "https" && url.userInfo == null && url.port in listOf(-1, 443) &&
                (url.host == "huggingface.co" || url.host.endsWith(".huggingface.co") || url.host == "hf.co" || url.host.endsWith(".hf.co"))) {
                "Небезопасный адрес скачивания модели"
            }
            val c = url.openConnection() as HttpsURLConnection
            c.instanceFollowRedirects = false
            c.connectTimeout = 30_000
            c.readTimeout = 30_000
            c.setRequestProperty("Accept-Encoding", "identity")
            try {
                val status = c.responseCode
                if (status in 300..399) {
                    val location = c.getHeaderField("Location") ?: error("Переадресация без адреса")
                    url = URL(url, location)
                    c.disconnect()
                } else {
                    if (status != HttpURLConnection.HTTP_OK) error("Ошибка загрузки HTTP $status")
                    return c
                }
            } catch (t: Throwable) {
                c.disconnect()
                throw t
            }
        }
        error("Слишком много переадресаций")
    }
}

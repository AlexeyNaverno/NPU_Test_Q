package ru.npu.llamachat.image

import android.content.Context
import java.io.File
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

    fun download(context: Context, model: ImageModel, checkCancelled: () -> Unit,
                 onFile: (String, Int, Int) -> Unit = { _, _, _ -> },
                 onRetry: (String, Int) -> Unit = { _, _ -> },
                 onVerify: (String) -> Unit = {},
                 progress: (Long, Long) -> Unit): File {
        require(model in ImageModelCatalog.models)
        recoverBackup(context, model)
        val stage = File(context.filesDir, ".${model.folder}-download-resume")
        check(stage.isDirectory || stage.mkdir()) { "Не удалось создать временную папку" }
        val remaining = model.files.sumOf { entry ->
            val file = File(stage, entry.path)
            if (file.isFile && file.length() <= entry.bytes) entry.bytes - file.length() else entry.bytes
        }
        require(context.filesDir.usableSpace >= remaining + 64L * 1024 * 1024) {
            "Недостаточно свободного места: необходимо ещё около ${(remaining / 1024 / 1024) + 64} МиБ"
        }
        var completed = 0L
        for ((index, entry) in model.files.withIndex()) {
            checkCancelled()
            onFile(entry.path, index + 1, model.files.size)
            val dest = File(stage, entry.path)
            check(dest.canonicalPath.startsWith(stage.canonicalPath + File.separator)) { "Неверный путь файла модели" }
            check(dest.parentFile?.let { it.isDirectory || it.mkdirs() } == true) { "Не удалось создать папку" }
            val base = completed
            progress(base + minOf(dest.length(), entry.bytes), model.totalBytes)
            val url = "https://huggingface.co/${model.repository}/resolve/${model.revision}/${entry.path}"
            ResumableFileDownload.download(entry, dest,
                ResumableFileDownload.Transport { start, end -> connect(url, start, end) }, checkCancelled,
                { current -> progress(base + current, model.totalBytes) },
                onRetry = { attempt, _ -> onRetry(entry.path, attempt) },
                onVerify = { onVerify(entry.path) })
            completed += entry.bytes
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
    }

    private fun connect(initialUrl: String, start: Long, end: Long): ResumableFileDownload.Response {
        var url = URL(initialUrl)
        repeat(6) {
            require(url.protocol == "https" && url.userInfo == null && url.port in listOf(-1, 443) &&
                (url.host == "huggingface.co" || url.host.endsWith(".huggingface.co") || url.host == "hf.co" || url.host.endsWith(".hf.co"))) {
                "Небезопасный адрес скачивания модели"
            }
            val c = url.openConnection() as HttpsURLConnection
            c.instanceFollowRedirects = false
            c.connectTimeout = 60_000
            c.readTimeout = 90_000
            c.setRequestProperty("Accept-Encoding", "identity")
            c.setRequestProperty("Range", "bytes=$start-$end")
            try {
                val httpStatus = c.responseCode
                if (httpStatus in 300..399) {
                    val location = c.getHeaderField("Location") ?: error("Переадресация без адреса")
                    url = URL(url, location)
                    c.disconnect()
                } else {
                    return object : ResumableFileDownload.Response {
                        override val status = httpStatus
                        override val range = c.getHeaderField("Content-Range")
                        override val encoding = c.getHeaderField("Content-Encoding")
                        override fun stream() = c.inputStream
                        override fun close() = c.disconnect()
                    }
                }
            } catch (t: Throwable) {
                c.disconnect()
                throw t
            }
        }
        error("Слишком много переадресаций")
    }
}

package ru.npu.llamachat.image

import org.json.JSONObject
import java.io.File

/** Чтение файлов CLIP из локального комплекта; проверяет размеры до загрузки в память. */
internal object ClipTokenizerFiles {
    private const val MAX_FILE_BYTES = 8L * 1024 * 1024

    fun load(directory: File): ClipBpeTokenizer {
        val vocabFile = checkedFile(directory, "vocab.json")
        val mergesFile = checkedFile(directory, "merges.txt")
        val json = JSONObject(vocabFile.readText(Charsets.UTF_8))
        val vocabulary = HashMap<String, Int>()
        val keys = json.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            val value = json.get(key)
            require(value is Int && value >= 0) { "Некорректный идентификатор токена" }
            vocabulary[key] = value
        }
        require(vocabulary.isNotEmpty() && vocabulary.size <= 100_000) { "Неверный размер словаря" }
        val merges = ArrayList<Pair<String, String>>()
        mergesFile.useLines(Charsets.UTF_8) { lines ->
            lines.forEach { raw ->
                val line = raw.trim()
                if (line.isNotEmpty() && !line.startsWith("#")) {
                    val parts = line.split(Regex("\\s+"))
                    require(parts.size == 2 && parts.all { it.isNotEmpty() }) { "Некорректное слияние BPE" }
                    merges.add(parts[0] to parts[1])
                    require(merges.size <= 100_000) { "Слишком много слияний BPE" }
                }
            }
        }
        return ClipBpeTokenizer(vocabulary, merges)
    }

    private fun checkedFile(directory: File, name: String): File {
        require(directory.isDirectory) { "Папка токенизатора не найдена" }
        val file = File(directory, name)
        require(file.isFile && file.canRead() && file.length() in 1..MAX_FILE_BYTES) {
            "Отсутствует или слишком велик файл $name"
        }
        return file
    }
}

package ru.npu.llamachat.image

import java.nio.charset.StandardCharsets
import java.util.Locale
import java.util.regex.Pattern

/**
 * Byte-level BPE для CLIP SD 1.5: словарь и список слияний предоставляются комплектом модели.
 * Не выполняет ftfy или HTML-unescape, поэтому для нестандартного ввода возможны отличия
 * от эталонного CLIP tokenizer. Не подключён к модели, совместимость необходимо проверить.
 */
internal class ClipBpeTokenizer(vocabulary: Map<String, Int>, merges: List<Pair<String, String>>) {
    private val vocabulary = vocabulary.toMap()
    private val ranks = merges.withIndex().associate { it.value to it.index }

    init {
        require(vocabulary[START] == BOS && vocabulary[END] == EOS) { "Неверные специальные токены CLIP" }
        require(ranks.size == merges.size) { "Повторяющееся слияние BPE" }
        require(vocabulary.values.all { it >= 0 })
    }

    fun tokenize(prompt: String): IntArray {
        require(prompt.length <= 4096) { "Слишком длинное описание" }
        val result = IntArray(MAX_LENGTH) { EOS }
        result[0] = BOS
        var length = 1
        val text = prompt.lowercase(Locale.ROOT).trim().replace(WHITESPACE, " ")
        val matcher = TOKEN_PATTERN.matcher(text)
        while (matcher.find() && length < MAX_LENGTH - 1) {
            val token = matcher.group()
            val encoded = buildString {
                token.toByteArray(StandardCharsets.UTF_8).forEach { append(BYTE_ENCODER[it.toInt() and 0xff]) }
            }
            val parts = if (encoded == START || encoded == END) listOf(encoded) else bpe(encoded)
            for (part in parts) {
                if (length == MAX_LENGTH - 1) break
                result[length++] = vocabulary[part] ?: throw IllegalArgumentException("Токен отсутствует в словаре CLIP")
            }
        }
        // Оставшийся массив заполнен EOS; на последней позиции EOS даже при усечении.
        return result
    }

    private fun bpe(token: String): List<String> {
        if (token.isEmpty()) return emptyList()
        val word = token.map { it.toString() }.toMutableList()
        word[word.lastIndex] += "</w>"
        while (word.size > 1) {
            var best: Pair<String, String>? = null
            var bestRank = Int.MAX_VALUE
            for (i in 0 until word.lastIndex) {
                val pair = word[i] to word[i + 1]
                val rank = ranks[pair] ?: continue
                if (rank < bestRank) { best = pair; bestRank = rank }
            }
            val merge = best ?: break
            val next = ArrayList<String>(word.size)
            var i = 0
            while (i < word.size) {
                if (i < word.lastIndex && word[i] == merge.first && word[i + 1] == merge.second) {
                    next.add(word[i] + word[i + 1]); i += 2
                } else {
                    next.add(word[i]); i++
                }
            }
            word.clear()
            word.addAll(next)
        }
        return word
    }

    companion object {
        const val MAX_LENGTH = 77
        const val BOS = 49406
        const val EOS = 49407
        private const val START = "<|startoftext|>"
        private const val END = "<|endoftext|>"
        private val WHITESPACE = Regex("\\s+")
        private val TOKEN_PATTERN = Pattern.compile(
            "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
            Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE or Pattern.UNICODE_CHARACTER_CLASS
        )
        private val BYTE_ENCODER: Array<String> = run {
            val bytes = mutableListOf<Int>()
            bytes.addAll(33..126); bytes.addAll(161..172); bytes.addAll(174..255)
            val codePoints = bytes.toMutableList()
            var next = 0
            for (b in 0..255) if (b !in bytes) { bytes.add(b); codePoints.add(256 + next++) }
            Array(256) { "" }.also { table ->
                for (i in bytes.indices) table[bytes[i]] = String(Character.toChars(codePoints[i]))
            }
        }
    }
}

package ru.npu.llamachat.model

/**
 * Эфемерное сообщение чата.
 * Хранится исключительно в оперативной памяти ViewModel и уничтожается при очистке чата или выходе из приложения.
 */
enum class MessageRole(val wireRole: String) {
    USER("user"),
    ASSISTANT("assistant")
}

data class ChatMessage(
    val id: Long,
    val role: MessageRole,
    val content: String,
    val tokenCount: Int = 0,
    val tokPerSec: Double = 0.0,
    val isStreaming: Boolean = false,
    val wasStopped: Boolean = false
)

package ru.npu.llamachat.api

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import ru.npu.llamachat.model.ChatMessage
import ru.npu.llamachat.model.MessageRole
import ru.npu.llamachat.ui.ChatViewModel
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.security.MessageDigest
import java.util.UUID
import kotlin.concurrent.thread

/** Одно соединение за раз, без сохранения запросов; только IPv4 loopback. */
class LocalApiServer(private val session: ChatViewModel, private val token: String) : AutoCloseable {
    private var listener: ServerSocket? = null
    @Volatile private var client: Socket? = null
    @Volatile private var closed = false

    fun start() {
        val socket = ServerSocket(8765, 8, InetAddress.getByName("127.0.0.1"))
        listener = socket
        thread(name = "npu-api", isDaemon = true) {
            while (!closed) {
                try {
                    socket.accept().use { connection ->
                        client = connection
                        connection.soTimeout = 10_000
                        try { handle(connection) } finally { client = null }
                    }
                } catch (_: Exception) { if (closed) break }
            }
        }
    }

    private fun line(input: BufferedInputStream): String {
        val bytes = ByteArrayOutputStream()
        while (bytes.size() < 8192) {
            val ch = input.read()
            require(ch >= 0) { "Неполный запрос" }
            if (ch == 10) return bytes.toString("UTF-8").trimEnd('\r')
            bytes.write(ch)
        }
        throw IllegalArgumentException("Слишком длинный заголовок")
    }

    private fun handle(socket: Socket) {
        try {
            val input = BufferedInputStream(socket.getInputStream())
            val request = line(input).split(' ')
            require(request.size == 3 && request[2].startsWith("HTTP/1.")) { "Неверный запрос" }
            val headers = mutableMapOf<String, String>()
            var total = 0
            while (true) {
                val h = line(input)
                if (h.isEmpty()) break
                total += h.length
                require(total < 16384) { "Слишком большие заголовки" }
                val colon = h.indexOf(':')
                require(colon > 0) { "Неверный заголовок" }
                val name = h.substring(0, colon).lowercase()
                require(name !in headers) { "Повторяющийся заголовок" }
                headers[name] = h.substring(colon + 1).trim()
            }
            val expected = "Bearer $token".toByteArray(Charsets.UTF_8)
            val auth = headers["authorization"].orEmpty().toByteArray(Charsets.UTF_8)
            if (!MessageDigest.isEqual(expected, auth)) { respond(socket, 401, error("Требуется Bearer token")); return }
            require(!headers.containsKey("transfer-encoding")) { "Transfer-Encoding не поддерживается" }
            if (request[0] == "GET" && request[1] == "/v1/models") {
                val model = session.uiState.value.loadedModel
                val data = JSONArray()
                if (model != null) data.put(JSONObject().put("id", model.fileName).put("object", "model").put("owned_by", "local"))
                respond(socket, 200, JSONObject().put("object", "list").put("data", data))
                return
            }
            if (request[0] != "POST" || request[1] != "/v1/chat/completions") {
                respond(socket, 404, error("Маршрут не найден")); return
            }
            val size = headers["content-length"]?.toIntOrNull() ?: throw IllegalArgumentException("Нужен Content-Length")
            require(size in 1..262144) { "Размер запроса вне допустимого диапазона" }
            val data = ByteArray(size)
            var offset = 0
            while (offset < size) {
                val n = input.read(data, offset, size - offset)
                require(n > 0) { "Неполное тело запроса" }
                offset += n
            }
            val json = JSONObject(String(data, Charsets.UTF_8))
            require(!json.optBoolean("stream", false)) { "Потоковый ответ пока не поддерживается" }
            val allowed = setOf("model", "messages", "stream")
            val keys = json.keys()
            while (keys.hasNext()) require(keys.next() in allowed) { "Параметр запроса не поддерживается" }
            val entries = json.getJSONArray("messages")
            require(entries.length() in 1..128) { "Количество сообщений вне допустимого диапазона" }
            var system: String? = null
            val messages = mutableListOf<ChatMessage>()
            for (i in 0 until entries.length()) {
                val item = entries.getJSONObject(i)
                val content = item.getString("content")
                require(content.isNotBlank()) { "Пустое сообщение" }
                when (item.getString("role")) {
                    "system" -> { require(i == 0 && system == null); system = content }
                    "user" -> messages.add(ChatMessage(i.toLong(), MessageRole.USER, content))
                    "assistant" -> messages.add(ChatMessage(i.toLong(), MessageRole.ASSISTANT, content))
                    else -> throw IllegalArgumentException("Неизвестная роль")
                }
            }
            require(messages.lastOrNull()?.role == MessageRole.USER) { "Последнее сообщение должно быть от пользователя" }
            val model = session.uiState.value.loadedModel?.fileName ?: throw IllegalStateException("Модель не загружена")
            require(!json.has("model") || json.getString("model") == model) { "Указана другая модель" }
            val (reply, stats) = runBlocking { withContext(Dispatchers.Main) { session.generateForApi(messages, system) } }
            val usage = JSONObject().put("prompt_tokens", stats.promptTokens).put("completion_tokens", stats.generatedTokens)
                .put("total_tokens", stats.promptTokens + stats.generatedTokens)
            val choice = JSONObject().put("index", 0).put("message", JSONObject().put("role", "assistant").put("content", reply))
                .put("finish_reason", "stop")
            respond(socket, 200, JSONObject().put("id", "chatcmpl-${UUID.randomUUID()}").put("object", "chat.completion")
                .put("created", System.currentTimeMillis() / 1000).put("model", model).put("choices", JSONArray().put(choice)).put("usage", usage))
        } catch (e: IllegalStateException) {
            runCatching { respond(socket, 503, error(e.message ?: "NPU недоступен")) }
        } catch (e: Exception) {
            runCatching { respond(socket, 400, error(e.message ?: "Неверный запрос")) }
        }
    }

    private fun error(message: String) = JSONObject().put("error", JSONObject().put("message", message))
    private fun respond(socket: Socket, status: Int, json: JSONObject) {
        val body = json.toString().toByteArray(Charsets.UTF_8)
        val reason = when (status) { 200 -> "OK"; 401 -> "Unauthorized"; 404 -> "Not Found"; 503 -> "Service Unavailable"; else -> "Bad Request" }
        val head = "HTTP/1.1 $status $reason\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: ${body.size}\r\nConnection: close\r\nCache-Control: no-store\r\n\r\n"
        socket.getOutputStream().apply { write(head.toByteArray(Charsets.US_ASCII)); write(body); flush() }
    }

    override fun close() {
        closed = true
        runCatching { listener?.close() }
        runCatching { client?.close() }
    }
}

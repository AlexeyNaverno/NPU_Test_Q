package ru.npu.llamachat.image

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.qnnpluginep.getEpName
import ai.onnxruntime.qnnpluginep.getLibraryPath
import android.content.Context
import android.system.Os
import ru.npu.llamachat.engine.SnapdragonNpuDetector
import java.io.Closeable
import java.io.File
import java.security.MessageDigest

/**
 * Экспериментальный загрузчик ONNX EPContext. Сам по себе не реализует генерацию.
 * Перед использованием вызывающий код обязан освободить LLM и сериализовать работу с HTP.
 * Не создаёт CPU/GPU сессий; отсутствие совместимого HTP или графа приводит к ошибке.
 */
object QnnHtpRuntime {
    private const val REGISTRATION_NAME = "QNNExecutionProvider"
    private var environment: OrtEnvironment? = null

    @Synchronized
    private fun getEnvironment(context: Context): OrtEnvironment {
        environment?.let { return it }
        val soc = SnapdragonNpuDetector.inspectCurrentDevice()
        check(soc.blockingReason == null) { soc.blockingReason ?: "Qualcomm Hexagon NPU недоступен" }
        val libraryDir = context.applicationContext.applicationInfo.nativeLibraryDir
        // До инициализации ORT/QNN: FastRPC должен видеть реальные файлы skel на диске.
        Os.setenv("ADSP_LIBRARY_PATH", libraryDir, true)
        val env = OrtEnvironment.getEnvironment()
        env.registerExecutionProviderLibrary(REGISTRATION_NAME, getLibraryPath())
        val devices = env.epDevices.filter { it.epName == getEpName() }
        if (devices.isEmpty()) {
            runCatching { env.unregisterExecutionProviderLibrary(REGISTRATION_NAME) }
            error("QNN не обнаружил устройство Hexagon HTP")
        }
        // Библиотека остаётся зарегистрированной до завершения процесса; выгружать её,
        // пока существует хоть одна сессия, опасно.
        environment = env
        return env
    }

    /**
     * Загружает один локальный ONNX EPContext с бинарным контекстом рядом с ним.
     * По умолчанию запрещает CPU fallback. Исключение для проверенных по SHA256
     * обёрток DreamShaper позволяет ORT назначить CPU только внешние узлы Q/DQ.
     */
    fun open(context: Context, modelFile: File, allowKnownQuantizationOnCpu: Boolean = false,
             symbolicDimensions: Map<String, Long> = emptyMap()): HtpSession {
        require(modelFile.isFile && modelFile.canRead() && modelFile.extension.equals("onnx", true)) {
            "Не найден читаемый ONNX файл: ${modelFile.name}"
        }
        if (allowKnownQuantizationOnCpu) requireKnownWrapper(modelFile)
        val env = getEnvironment(context)
        val devices = env.epDevices.filter { it.epName == getEpName() }
        check(devices.isNotEmpty()) { "Устройство QNN HTP недоступно" }
        val options = OrtSession.SessionOptions()
        try {
            // В закреплённых обёртках вне EPContext есть только Q/DQ. CPU разрешён
            // лишь для них; сам EPContext может выполняться только через QNN HTP.
            if (!allowKnownQuantizationOnCpu) options.addConfigEntry("session.disable_cpu_ep_fallback", "1")
            symbolicDimensions.forEach { (name, value) ->
                require(name.isNotBlank() && value > 0)
                options.setSymbolicDimensionValue(name, value)
            }
            options.addExecutionProvider(devices, mapOf("backend_type" to "htp"))
            val session = env.createSession(modelFile.absolutePath, options)
            return HtpSession(session, options)
        } catch (t: Throwable) {
            options.close()
            throw t
        }
    }

    private fun requireKnownWrapper(file: File) {
        val hashes = mapOf(
            "text_encoder.onnx" to "fc06df873065d0129fcb1e42b85200486bc9414b6914f2f82ea69a8c8ed6c8b8",
            "unet.onnx" to "286a49715a74e7fc739ae972f27965b70458d62130b3f83e1c28d65eec30121b",
            "vae.onnx" to "a92718467028e6289be68e46e7000f5e99d0d14377ebfbec738cee8a05bef29d"
        )
        val expected = hashes[file.name] ?: throw IllegalArgumentException("Неизвестная обёртка QNN")
        require(file.length() in 1..4096) { "Неверный размер обёртки QNN" }
        val actual = MessageDigest.getInstance("SHA-256").digest(file.readBytes())
            .joinToString("") { "%02x".format(it.toInt() and 255) }
        require(actual == expected) { "Обёртка QNN не соответствует проверенной ревизии" }
    }

    class HtpSession internal constructor(
        val session: OrtSession,
        private val options: OrtSession.SessionOptions
    ) : Closeable {
        override fun close() {
            try { session.close() } finally { options.close() }
        }
    }
}

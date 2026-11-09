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
     * Строгая настройка запрещает ORT запускать неподдерживаемые операторы на CPU.
     */
    fun open(context: Context, modelFile: File): HtpSession {
        require(modelFile.isFile && modelFile.canRead() && modelFile.extension.equals("onnx", true)) {
            "Не найден читаемый ONNX файл: ${modelFile.name}"
        }
        val env = getEnvironment(context)
        val devices = env.epDevices.filter { it.epName == getEpName() }
        check(devices.isNotEmpty()) { "Устройство QNN HTP недоступно" }
        val options = OrtSession.SessionOptions()
        try {
            options.addConfigEntry("session.disable_cpu_ep_fallback", "1")
            options.addExecutionProvider(devices, mapOf("backend_type" to "htp"))
            val session = env.createSession(modelFile.absolutePath, options)
            return HtpSession(session, options)
        } catch (t: Throwable) {
            options.close()
            throw t
        }
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

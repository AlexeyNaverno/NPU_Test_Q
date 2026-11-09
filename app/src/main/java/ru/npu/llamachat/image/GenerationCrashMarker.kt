package ru.npu.llamachat.image

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context

/** Сохраняет только этап работы и время, никогда не сохраняет текст промпта или изображение. */
internal class GenerationCrashMarker(private val context: Context) {
    private val prefs = context.getSharedPreferences("image_generation_diagnostics", Context.MODE_PRIVATE)

    fun start() {
        prefs.edit().putLong("started", System.currentTimeMillis()).putString("stage", "подготовка").commit()
    }

    fun stage(stage: String) {
        if (prefs.getLong("started", 0) > 0) prefs.edit().putString("stage", stage).commit()
    }

    fun finish() { prefs.edit().clear().commit() }

    /** Однократно сообщает о прерывании предыдущего процесса; история текста не читается. */
    fun takePreviousExit(): String? {
        val started = prefs.getLong("started", 0)
        if (started <= 0) return null
        val stage = prefs.getString("stage", "подготовка").orEmpty()
        val exit = runCatching {
            context.getSystemService(ActivityManager::class.java)
                ?.getHistoricalProcessExitReasons(context.packageName, 0, 16)
                ?.filter { it.timestamp >= started && it.processName == context.packageName }
                ?.maxByOrNull { it.timestamp }
        }.getOrNull()
        finish()
        val reason = when (exit?.reason) {
            ApplicationExitInfo.REASON_CRASH_NATIVE -> "нативный сбой"
            ApplicationExitInfo.REASON_CRASH -> "ошибка приложения"
            ApplicationExitInfo.REASON_LOW_MEMORY -> "нехватка памяти"
            ApplicationExitInfo.REASON_ANR -> "приложение не отвечало"
            ApplicationExitInfo.REASON_SIGNALED -> "процесс завершён сигналом"
            ApplicationExitInfo.REASON_USER_REQUESTED -> "приложение остановлено пользователем или системой"
            else -> "причина не определена"
        }
        return "Предыдущая генерация прервалась. Этап: $stage; $reason. Сообщите этот текст для диагностики."
    }
}

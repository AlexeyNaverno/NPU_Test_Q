package ru.npu.llamachat.api

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Base64
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import ru.npu.llamachat.MainActivity
import ru.npu.llamachat.NpuApplication
import ru.npu.llamachat.R
import java.security.SecureRandom

class ApiService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var server: LocalApiServer? = null
    private val session get() = (application as NpuApplication).session

    override fun onCreate() {
        super.onCreate()
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel(CHANNEL, "Локальный API NPU", NotificationManager.IMPORTANCE_LOW))
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val load = PendingIntent.getActivity(this, 3, Intent(this, MainActivity::class.java).setAction(LOAD)
            .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP), PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val stop = PendingIntent.getService(this, 1, Intent(this, ApiService::class.java).setAction(STOP), PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val unload = PendingIntent.getService(this, 2, Intent(this, ApiService::class.java).setAction(UNLOAD), PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val notification = NotificationCompat.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_npu_chip)
            .setContentTitle("Локальный API NPU")
            .setContentText("Сервис активен; API доступен только на этом устройстве")
            .setContentIntent(open).setOngoing(true)
            .addAction(0, "Выбрать модель", load)
            .addAction(0, "Выгрузить модель", unload)
            .addAction(0, "Остановить API", stop)
            .build()
        startForeground(31, notification)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            STOP -> { stopSelf(); return START_NOT_STICKY }
            UNLOAD -> {
                session.cancelActiveTask()
                scope.launch {
                    session.uiState.first { !it.isApiGenerating && !it.isLoadingModel }
                    session.unloadModel()
                }
                return START_NOT_STICKY
            }
        }
        if (server == null) {
            val bytes = ByteArray(32).also { SecureRandom().nextBytes(it) }
            val token = Base64.encodeToString(bytes, Base64.NO_WRAP or Base64.URL_SAFE or Base64.NO_PADDING)
            try {
                server = LocalApiServer(session, token).also { it.start() }
                apiToken.value = token
                error.value = ""
                running.value = true
            } catch (e: Exception) {
                error.value = "Не удалось запустить API: ${e.message}"
                stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        if (session.uiState.value.isApiGenerating) session.cancelActiveTask()
        server?.close()
        server = null
        running.value = false
        apiToken.value = ""
        scope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
    companion object {
        private const val CHANNEL = "npu_local_api"
        const val LOAD = "ru.npu.llamachat.SELECT_MODEL"
        const val STOP = "ru.npu.llamachat.STOP_API"
        const val UNLOAD = "ru.npu.llamachat.UNLOAD_MODEL"
        val running = MutableStateFlow(false)
        val apiToken = MutableStateFlow("")
        val error = MutableStateFlow("")
    }
}

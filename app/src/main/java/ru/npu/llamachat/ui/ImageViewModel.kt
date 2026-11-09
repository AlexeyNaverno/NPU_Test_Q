package ru.npu.llamachat.ui

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import ru.npu.llamachat.NpuApplication
import ru.npu.llamachat.image.DreamShaperLcm
import ru.npu.llamachat.image.DreamShaperPackageImporter
import kotlin.math.roundToInt

data class ImageUiState(
    val busy: Boolean = false,
    val installed: Boolean = false,
    val status: String = "Выберите папку модели DreamShaper 8 LCM v73",
    val bitmap: Bitmap? = null
)

class ImageViewModel(application: Application) : AndroidViewModel(application) {
    private val app = application as NpuApplication
    private val _state = MutableStateFlow(ImageUiState(installed = DreamShaperPackageImporter.installedDirectory(app).isDirectory))
    val state = _state.asStateFlow()
    private var job: Job? = null

    fun importModel(uri: Uri) {
        if (_state.value.busy) return
        _state.update { it.copy(busy = true, status = "Импорт модели…") }
        job = viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    val context = currentCoroutineContext()
                    var previousPercent = -1
                    DreamShaperPackageImporter.import(app, uri, { context.ensureActive() }) { bytes, total ->
                        val percent = (bytes * 100 / total).toInt()
                        if (percent != previousPercent) {
                            previousPercent = percent
                            _state.update { it.copy(status = "Импорт модели: $percent%") }
                        }
                    }
                }
                _state.update { it.copy(installed = true, status = "Модель импортирована. Выгрузите модель чата перед генерацией.") }
            } catch (e: CancellationException) {
                _state.update { it.copy(status = "Импорт отменён") }
            } catch (e: Exception) {
                _state.update { it.copy(status = "Ошибка импорта: ${e.message ?: "неизвестная ошибка"}") }
            } finally {
                _state.update { it.copy(busy = false) }
            }
        }
    }

    fun generate(prompt: String, steps: Int, seed: Long) {
        if (_state.value.busy) return
        if (!_state.value.installed) { _state.update { it.copy(status = "Сначала импортируйте модель") }; return }
        if (prompt.isBlank()) { _state.update { it.copy(status = "Введите описание изображения") }; return }
        if (!app.session.tryAcquireImageNpu()) {
            _state.update { it.copy(status = "NPU занят или недоступен. Сначала остановите и выгрузите модель чата.") }
            return
        }
        _state.update { it.copy(busy = true, status = "Подготовка графов QNN…", bitmap = null) }
        job = viewModelScope.launch {
            try {
                val bitmap = withContext(Dispatchers.IO) {
                    val context = currentCoroutineContext()
                    val rgb = DreamShaperLcm.generate(app, DreamShaperPackageImporter.installedDirectory(app), prompt, steps, seed,
                        checkCancelled = { context.ensureActive() },
                        onStep = { current, total -> _state.update { it.copy(status = "Шаг $current из $total") } })
                    context.ensureActive()
                    val colors = IntArray(512 * 512) { i ->
                        val r = (rgb[i * 3] * 255f).roundToInt().coerceIn(0, 255)
                        val g = (rgb[i * 3 + 1] * 255f).roundToInt().coerceIn(0, 255)
                        val b = (rgb[i * 3 + 2] * 255f).roundToInt().coerceIn(0, 255)
                        (0xff shl 24) or (r shl 16) or (g shl 8) or b
                    }
                    Bitmap.createBitmap(colors, 512, 512, Bitmap.Config.ARGB_8888)
                }
                _state.update { it.copy(bitmap = bitmap, status = "Готово. Seed: $seed") }
            } catch (e: CancellationException) {
                _state.update { it.copy(status = "Генерация отменена") }
            } catch (e: Exception) {
                _state.update { it.copy(status = "Ошибка генерации: ${e.message ?: "неизвестная ошибка"}") }
            } finally {
                app.session.releaseImageNpu()
                _state.update { it.copy(busy = false) }
            }
        }
    }

    /** Отмена применяется между вызовами графов; выполняющийся вызов QNN немедленно не прерывается. */
    fun cancel() { job?.cancel() }
}

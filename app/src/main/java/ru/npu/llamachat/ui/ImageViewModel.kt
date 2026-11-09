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
import ru.npu.llamachat.image.AnythingV5
import ru.npu.llamachat.image.DreamShaperPackageImporter
import ru.npu.llamachat.image.ImageModelCatalog
import ru.npu.llamachat.image.ImageModelDownloader
import kotlin.math.roundToInt

data class ImageUiState(
    val busy: Boolean = false,
    val selectedModel: Int = 0,
    val canGenerate: Boolean = true,
    val installed: Boolean = false,
    val status: String = "Выберите папку модели DreamShaper 8 LCM v73",
    val bitmap: Bitmap? = null
)

class ImageViewModel(application: Application) : AndroidViewModel(application) {
    private val app = application as NpuApplication
    private val _state = MutableStateFlow(ImageUiState(installed = ImageModelDownloader.isInstalled(app, ImageModelCatalog.dreamShaper)))
    val state = _state.asStateFlow()
    private var job: Job? = null

    fun selectModel(index: Int) {
        if (_state.value.busy || index !in ImageModelCatalog.models.indices) return
        val model = ImageModelCatalog.models[index]
        _state.update { it.copy(selectedModel = index, canGenerate = model.canGenerate,
            installed = ImageModelDownloader.isInstalled(app, model),
            status = if (index == 0) "DreamShaper LCM: рекомендуются 4 шага; совместимость NPU не проверена."
            else "Anything V5: экспериментальная попытка HTP для UNet/VAE, CLIP на CPU. Исходный комплект рассчитан на CPU и может не запуститься на NPU.") }
    }

    fun downloadModel() {
        if (_state.value.busy) return
        val model = ImageModelCatalog.models[_state.value.selectedModel]
        _state.update { it.copy(busy = true, status = "Подготовка загрузки…") }
        job = viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    val context = currentCoroutineContext()
                    var previousPercent = -1
                    var currentFile = ""
                    ImageModelDownloader.download(app, model, { context.ensureActive() },
                        onFile = { name, index, count ->
                            currentFile = "Файл $index/$count: $name"
                            previousPercent = -1
                            _state.update { it.copy(status = currentFile) }
                        },
                        onRetry = { name, attempt -> _state.update { it.copy(status = "Сбой сети, повтор $attempt/4: $name") } },
                        onVerify = { name -> _state.update { it.copy(status = "Проверка файла: $name") } }
                    ) { bytes, total ->
                        val percent = (bytes * 100 / total).toInt()
                        if (percent != previousPercent) {
                            previousPercent = percent
                            _state.update { it.copy(status = "$currentFile — $percent%") }
                        }
                    }
                }
                _state.update { it.copy(installed = true, status = "Модель скачана. Выгрузите модель чата перед генерацией.") }
            } catch (e: CancellationException) {
                _state.update { it.copy(status = "Загрузка приостановлена; нажмите «Скачать» для продолжения") }
            } catch (e: Exception) {
                _state.update { it.copy(status = "Ошибка загрузки: ${e.message ?: "неизвестная ошибка"}. Данные сохранены; нажмите «Скачать» для продолжения.") }
            } finally {
                _state.update { it.copy(busy = false, installed = ImageModelDownloader.isInstalled(app, model)) }
            }
        }
    }

    fun deleteModel() {
        if (_state.value.busy) return
        val model = ImageModelCatalog.models[_state.value.selectedModel]
        _state.update { it.copy(busy = true, status = "Удаление модели…") }
        job = viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    val root = ImageModelDownloader.directory(app, model)
                    check(!root.exists() || root.deleteRecursively()) { "Не удалось удалить модель" }
                    val prefixes = listOf(".${model.folder}-download-", ".${model.folder}-old-")
                    app.filesDir.listFiles()?.filter { f -> prefixes.any { f.name.startsWith(it) } }
                        ?.forEach { check(it.deleteRecursively()) { "Не удалось удалить временные файлы" } }
                }
                _state.update { it.copy(installed = false, status = "Модель удалена") }
            } catch (e: Exception) {
                _state.update { it.copy(status = "Ошибка удаления: ${e.message ?: "неизвестная ошибка"}") }
            } finally {
                _state.update { it.copy(busy = false, installed = ImageModelDownloader.isInstalled(app, model)) }
            }
        }
    }

    fun importModel(uri: Uri) {
        if (_state.value.busy || _state.value.selectedModel != 0) return
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

    fun generate(prompt: String, negativePrompt: String, steps: Int, guidance: Float, seed: Long) {
        if (_state.value.busy) return
        if (!_state.value.canGenerate) { _state.update { it.copy(status = "Генерация Anything V5 пока не подключена") }; return }
        val model = ImageModelCatalog.models[_state.value.selectedModel]
        val maxSteps = if (model == ImageModelCatalog.dreamShaper) 50 else 100
        if (steps !in 1..maxSteps || !guidance.isFinite() || guidance !in 1f..20f) {
            _state.update { it.copy(status = "Допустимо шагов: 1–$maxSteps; CFG: 1–20") }; return
        }
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
                    val check = { context.ensureActive() }
                    val onStep: (Int, Int) -> Unit = { current, total -> _state.update { it.copy(status = "Шаг $current из $total") } }
                    val rgb = if (model == ImageModelCatalog.dreamShaper) {
                        DreamShaperLcm.generate(app, ImageModelDownloader.directory(app, model), prompt, steps, seed, check, onStep)
                    } else {
                        AnythingV5.generate(app, ImageModelDownloader.directory(app, model), prompt, negativePrompt,
                            steps, guidance, seed, check, onStep)
                    }
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

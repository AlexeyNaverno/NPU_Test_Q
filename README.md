# NPU Llama Chat (`v0.0.1`)

[![Latest Release](https://img.shields.io/github/v/release/AlexeyNaverno/NPU_Test_Q?label=%D0%9F%D0%BE%D1%81%D0%BB%D0%B5%D0%B4%D0%BD%D0%B8%D0%B9%20%D1%80%D0%B5%D0%BB%D0%B8%D0%B7&style=for-the-badge)](https://github.com/AlexeyNaverno/NPU_Test_Q/releases/latest)

Минималистичное Android-приложение на базе **llama.cpp (`ggml-hexagon`)** для запуска локальных языковых моделей формата **`.gguf`** строго на нейропроцессоре **Qualcomm Hexagon NPU** (целевая платформа — **Snapdragon 8s Gen 3 / `SM8635` / Hexagon HTP v73** и совместимые чипсеты линейки Snapdragon).

## Скачать последний релиз

- **Страница последнего релиза**: [**github.com/AlexeyNaverno/NPU_Test_Q/releases/latest**](https://github.com/AlexeyNaverno/NPU_Test_Q/releases/latest) (текущая версия: [`v0.0.1`](https://github.com/AlexeyNaverno/NPU_Test_Q/releases/tag/v0.0.1))
- **Установочный Release APK (`arm64-v8a`, Hexagon NPU)**: [**Скачать `NPU_Llama_Chat-v0.0.1-release.apk`**](https://github.com/AlexeyNaverno/NPU_Test_Q/releases/latest/download/NPU_Llama_Chat-v0.0.1-release.apk)
- **Debug APK и контрольные суммы**: [`NPU_Llama_Chat-v0.0.1-debug.apk`](https://github.com/AlexeyNaverno/NPU_Test_Q/releases/latest/download/NPU_Llama_Chat-v0.0.1-debug.apk) • [`SHA256SUMS.txt`](https://github.com/AlexeyNaverno/NPU_Test_Q/releases/latest/download/SHA256SUMS.txt)

---

## Ключевые возможности релиза `0.0.1`

1. **Строгое выполнение на NPU (Zero CPU/GPU Fallback)**
   - Использует официальный аппаратный бэкенд `ggml-hexagon` (`HTP0` через Qualcomm FastRPC `cdsprpcd` с векторными расширениями HVX и матричным блоком HMX).
   - Поддерживает **Snapdragon 8s Gen 3 (`SM8635`, Hexagon HTP v73)**, а также совместимые SoC с Hexagon HTP:
     - `v73`: Snapdragon 8s Gen 3 (`SM8635`), Snapdragon 7+ Gen 3 (`SM7675`), Snapdragon 8 Gen 2 (`SM8550`)
     - `v75`: Snapdragon 8 Gen 3 (`SM8650`)
     - `v79`: Snapdragon 8s Gen 4 (`SM8735`), Snapdragon 8 Elite (`SM8750`)
     - `v81` / `v85`: новейшие поколения Snapdragon 8 Elite
   - **Запрет запуска чисто на CPU или GPU**:
     1. На этапе инициализации проверяется наличие чипсета Qualcomm arm64 и активного FastRPC-устройства `HTP0`.
     2. В `llama_model_params.devices` передаётся исключительно массив указателей `{ HTP0, nullptr }` (CPU и Adreno OpenCL исключены).
     3. Используется `LLAMA_LOAD_MODE_NONE` (`--no-mmap`) для прямой переупаковки тензоров в тайлы HMX/HVX в памяти FastRPC RPCMEM (`n_gpu_layers >= 1`, по умолчанию все слои `999`).
     4. После чтения `.gguf` проверяется фактическая выгрузка слоёв и ненулевой размер буфера `HTP0 model buffer size > 0 MiB`; если ни один слой не размещён на NPU, модель немедленно выгружается с ошибкой.

2. **Выбор любого `.gguf` файла с устройства**
   - Загрузка через системный пикер Android Storage Access Framework (`OpenDocument`) без запроса разрешений на чтение всего хранилища и без дублирования многогигабайтного `.gguf` файла во внутреннюю папку приложения.
   - Проверка заголовка `GGUF` (версии v2 / v3) и прогресс-бар загрузки весов в NPU.

3. **Минимальный чат без сохранения истории + очистка чата**
   - Потоковая генерация с посимвольной защитой многобайтовых последовательностей UTF-8 (русская кириллица и 4-байтовые эмодзи) и переиспользованием префикса KV-кэша между репликами диалога.
   - **Приватность по умолчанию**: сообщения хранятся исключительно в оперативной памяти `ChatViewModel`, не записываются в БД/файлы/`SharedPreferences`, автобэкап отключён (`android:allowBackup="false"`), для клавиатуры выставлен `flagNoPersonalizedLearning`.
   - Кнопка **«Очистить чат»** мгновенно останавливает активную генерацию, очищает список сообщений, сбрасывает счётчики и очищает KV-кэш на NPU (`llama_memory_clear`).

4. **Подсчёт токенов и мониторинг ОЗУ**
   - Живой счётчик токенов контекста (`использовано / максимум`), токенов промпта, попаданий в KV-кэш, сгенерированных токенов ответа, скорости декодирования (`ток/с`) и токенов черновика в поле ввода.
   - Непрерывный мониторинг физической памяти процесса (`VmRSS` и пик `VmHWM` из `/proc/self/status`), размера NPU-буфера `HTP0` и свободной оперативной памяти смартфона.

5. **Гибкая конфигурация модели**
   - Настройка максимального размера контекста (`n_ctx` от `512` до `32768`), лимита новых токенов, размера батча (`n_batch`), температуры, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `seed`, системного промпта, Flash Attention, типа KV-кэша (`F16` или экономичный `Q8_0`) и профиля питания Hexagon cDSP (`sustained_high_performance`, `burst`, `balanced`, `power_saver`).

---

## Рекомендуемые модели `.gguf` для Snapdragon 8s Gen 3 (Hexagon HTP v73)

Бэкенд `ggml-hexagon` наиболее эффективно ускоряет матричные операции в форматах **`Q4_0`**, **`Q8_0`**, **`Q4_K_M`**, **`IQ4_NL`** и **`F16`**:
- **Для смартфонов с 8 ГБ ОЗУ**: модели 1B–3B (`Qwen2.5-3B-Instruct-Q4_0.gguf`, `Llama-3.2-3B-Instruct-Q4_0.gguf`, `Gemma-3-4B-IT-Q4_0.gguf`) с контекстом `2048–4096`.
- **Для смартфонов с 12–16 ГБ ОЗУ**: модели 3B–8B (`Qwen2.5-7B-Instruct-Q4_0.gguf`, `Llama-3.1-8B-Instruct-Q4_0.gguf`) с контекстом `4096–8192` (при длинном контексте рекомендуется включить `Q8_0 KV-кэш` в настройках).

---

## Сборка через Gradle

### Требования для сборки
- **JDK 17**
- **Android SDK** (`compileSdk = 36`, **NDK `28.2.13676358`**, **CMake `3.31.6`**)

### Быстрая сборка APK (автозагрузка официального пакета `llama.cpp` Snapdragon Hexagon NPU)
Задача Gradle `prepareSnapdragonNpuSdk` автоматически скачивает и проверяет официальный архив `llama-b12211-bin-android-arm64-snapdragon.tar.gz` (включающий `libggml-hexagon.so` и DSP-модули `libggml-htp-v73.so` .. `v85.so`):

```bash
# Запуск модульных тестов
./gradlew testDebugUnitTest

# Сборка Debug APK
./gradlew assembleDebug

# Сборка подписанного Release APK v0.0.1
./gradlew assembleRelease
```

Готовый установочный APK будет создан по пути:
`app/build/outputs/apk/release/app-release.apk`

### Опционально: полная сборка Hexagon NPU SDK из исходников `llama.cpp`
Если требуется самостоятельно пересобрать `libllama.so`, `libggml-hexagon.so` и `libggml-htp-v73.so` с помощью Qualcomm Hexagon SDK (контейнер `ghcr.io/snapdragon-toolchain/arm64-android:v0.8`):

```bash
chmod +x scripts/build-hexagon-sdk.sh
./scripts/build-hexagon-sdk.sh
./gradlew assembleRelease
```

Либо можно указать путь к заранее распакованному Snapdragon SDK через свойство Gradle:
```bash
./gradlew assembleRelease -PllamaSnapdragonSdkDir=/path/to/llama-snapdragon-sdk
```

---

## Структура проекта

```text
├── app/
│   ├── build.gradle.kts                  # Конфигурация Gradle + задача prepareSnapdragonNpuSdk
│   ├── proguard-rules.pro                # Правила R8 для сохранения JNI-моста в Release
│   └── src/
│       ├── main/
│       │   ├── AndroidManifest.xml       # Доступ к libcdsprpc.so, extractNativeLibs=true, allowBackup=false
│       │   ├── cpp/
│       │   │   ├── CMakeLists.txt        # Сборка libnpu_llama_jni.so и проверка наличия libggml-htp-v73.so
│       │   │   ├── npu_engine.hpp        # Интерфейс C++ движка Hexagon NPU
│       │   │   ├── npu_engine.cpp        # Жёсткая привязка HTP0, KV-кэш, UTF-8 стриминг, мониторинг VmRSS
│       │   │   └── npu_llama_jni.cpp     # JNI-мост (UTF-16 <-> UTF-8 без ошибок CheckJNI)
│       │   ├── java/ru/npu/llamachat/
│       │   │   ├── MainActivity.kt       # Экран чата, SAF-пикер .gguf, панель токенов и ОЗУ
│       │   │   ├── engine/               # SnapdragonNpuDetector, NpuLlamaBridge, NpuChatEngine
│       │   │   ├── model/                # ModelConfig, ChatMessage, NpuTelemetry
│       │   │   └── ui/                   # ChatViewModel, ChatAdapter, SettingsBottomSheet
│       │   └── res/                      # Material 3 XML-разметка и ресурсы
│       └── test/                         # Модульные JUnit-тесты детектора SoC и конфигурации
├── scripts/
│   └── build-hexagon-sdk.sh              # Скрипт сборки Hexagon SDK из исходников в Docker
├── .github/workflows/
│   └── android-release.yml               # CI/CD сборка и проверка Release APK v0.0.1
├── CHANGELOG.md
└── README.md
```

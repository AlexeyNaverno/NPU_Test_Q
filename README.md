# NPU Llama Chat (`v0.0.2`)

[![Latest Release](https://img.shields.io/github/v/release/AlexeyNaverno/NPU_Test_Q?label=%D0%9F%D0%BE%D1%81%D0%BB%D0%B5%D0%B4%D0%BD%D0%B8%D0%B9%20%D1%80%D0%B5%D0%BB%D0%B8%D0%B7&style=for-the-badge)](https://github.com/AlexeyNaverno/NPU_Test_Q/releases/latest)

Минималистичное Android-приложение на базе **llama.cpp (`ggml-hexagon`)** для запуска локальных языковых моделей формата **`.gguf`** строго на нейропроцессоре **Qualcomm Hexagon NPU** (целевая платформа — **Snapdragon 8s Gen 3 / `SM8635` / Hexagon HTP v73** и совместимые чипсеты линейки Snapdragon).

## Скачать последний релиз

- **Страница последнего релиза**: [**github.com/AlexeyNaverno/NPU_Test_Q/releases/latest**](https://github.com/AlexeyNaverno/NPU_Test_Q/releases/latest) (текущая версия: [`v0.0.2`](https://github.com/AlexeyNaverno/NPU_Test_Q/releases/tag/v0.0.2))
- **Установочный Release APK (`arm64-v8a`, Hexagon NPU)**: [**Скачать `NPU_Llama_Chat-v0.0.2-release.apk`**](https://github.com/AlexeyNaverno/NPU_Test_Q/releases/latest/download/NPU_Llama_Chat-v0.0.2-release.apk)
- **Debug APK и контрольные суммы**: [`NPU_Llama_Chat-v0.0.2-debug.apk`](https://github.com/AlexeyNaverno/NPU_Test_Q/releases/latest/download/NPU_Llama_Chat-v0.0.2-debug.apk) • [`SHA256SUMS.txt`](https://github.com/AlexeyNaverno/NPU_Test_Q/releases/latest/download/SHA256SUMS.txt)

---

## Что нового в `v0.0.2`

- **Исправлен вывод «мусора» на `Qwen3-0.6B` (`Qwen 0.6B`) и субфлагманских чипах Hexagon HTP v73 (`SM8635` / `SM7675` / `SM7750`)**:
  - Автоматический выбор HVX-безопасного пути NPU (`GGML_HEXAGON_NHMX=0`, `MM_SELECT=1`) на SoC без аппаратной поддержки FP16-HMX, при сохранении 100% выгрузки слоёв на `HTP0`.
  - Включено ядро HVX Flash Attention (`GGML_HEXAGON_FA_SELECT=1`), устраняющее порчу DMA-масок GQA после 128 токенов, и фильтр `GGML_HEXAGON_OPFILTER=GET_ROWS` для корректного чтения таблиц токенов в `Q4_K`/`Q5_K`.
  - Полная поддержка шаблонов **Qwen3 / Qwen3.5** с блоком `<think>...</think>`, переключателем режима размышлений в настройках и командами `/think` и `/no_think`.
- **Устранён вылет приложения на моделях `9B` и других сверхкрупных `.gguf` файлах**:
  - Двухэтапная проверка бюджета 32-битного адресного пространства Hexagon cDSP (~`3.0–3.2 ГБ` на сессию `HTP0`) и свободной ОЗУ смартфона (`no_alloc = true`) **до** аллокации буферов FastRPC RPCMEM.
  - Вместо падения процесса от `rpcmem_alloc2` / Android `lmkd` выводится подробное окно диагностики с расчётом памяти и рекомендацией подходящего размера модели.

---

## Ключевые возможности

1. **Строгое выполнение на NPU (Zero CPU/GPU Fallback)**
   - Использует официальный аппаратный бэкенд `ggml-hexagon` (`HTP0` через Qualcomm FastRPC `cdsprpcd` с векторными расширениями HVX и матричным блоком HMX).
   - Поддерживает **Snapdragon 8s Gen 3 (`SM8635`, Hexagon HTP v73, режим HVX)**, а также совместимые SoC с Hexagon HTP:
     - `v73`: Snapdragon 8s Gen 3 (`SM8635`), Snapdragon 7+ Gen 3 (`SM7675`), Snapdragon 7 Gen 4 (`SM7750`), Snapdragon 8 Gen 2 (`SM8550`)
     - `v75`: Snapdragon 8 Gen 3 (`SM8650`)
     - `v79`: Snapdragon 8s Gen 4 (`SM8735`), Snapdragon 8 Elite (`SM8750`)
     - `v81` / `v85`: новейшие поколения Snapdragon 8 Elite
   - **Запрет запуска чисто на CPU или GPU**:
     1. На этапе инициализации проверяется наличие чипсета Qualcomm arm64 и активного FastRPC-устройства `HTP0`.
     2. В `llama_model_params.devices` передаётся исключительно массив указателей `{ HTP0, nullptr }` (CPU и Adreno OpenCL исключены).
     3. Используется `LLAMA_LOAD_MODE_NONE` (`--no-mmap`) для прямой переупаковки тензоров в тайлы NPU в памяти FastRPC RPCMEM (`n_gpu_layers >= 1`, по умолчанию все слои `999`).
     4. После чтения `.gguf` проверяется фактическая выгрузка слоёв и ненулевой размер буфера `HTP0 model buffer size > 0 MiB`; если ни один слой не размещён на NPU, модель немедленно выгружается с ошибкой.

2. **Выбор `.gguf` файла с предварительной защитой памяти NPU**
   - Загрузка через системный пикер Android Storage Access Framework (`OpenDocument`) без запроса разрешений на чтение всего хранилища и без дублирования `.gguf` файла во внутреннюю папку приложения.
   - Двухступенчатая проверка заголовка и тензорной карты `GGUF` до выделения DMA/RPCMEM буферов защищает приложение от аварийного завершения (`SIGKILL` / `OOM`) при случайном выборе слишком тяжёлой модели (`7B–9B+`).

3. **Минимальный чат без сохранения истории + очистка чата**
   - Потоковая генерация с посимвольной защитой многобайтовых последовательностей UTF-8 (русская кириллица и 4-байтовые эмодзи) и переиспользованием префикса KV-кэша между репликами диалога.
   - **Приватность по умолчанию**: сообщения хранятся исключительно в оперативной памяти `ChatViewModel`, не записываются в БД/файлы/`SharedPreferences`, автобэкап отключён (`android:allowBackup="false"`), для клавиатуры выставлен `flagNoPersonalizedLearning`.
   - Кнопка **«Очистить чат»** мгновенно останавливает активную генерацию, очищает список сообщений, сбрасывает счётчики и очищает KV-кэш на NPU (`llama_memory_clear`).

4. **Подсчёт токенов и мониторинг ОЗУ**
   - Живой счётчик токенов контекста (`использовано / максимум`), токенов промпта, попаданий в KV-кэш, сгенерированных токенов ответа, скорости декодирования (`ток/с`) и токенов черновика в поле ввода.
   - Непрерывный мониторинг физической памяти процесса (`VmRSS` и пик `VmHWM` из `/proc/self/status`), размера NPU-буфера `HTP0` и свободной оперативной памяти смартфона.

5. **Гибкая конфигурация модели и режима `<think>`**
   - Настройка максимального размера контекста (`n_ctx` от `512` до `32768`), лимита новых токенов, размера батча (`n_batch`), температуры, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `seed`, системного промпта, переключателя размышлений `<think>` (Qwen3 / DeepSeek R1), Flash Attention, типа KV-кэша (`F16` или экономичный `Q8_0`) и профиля питания Hexagon cDSP (`sustained_high_performance`, `burst`, `balanced`, `power_saver`).

---

## Рекомендуемые модели `.gguf` и аппаратный лимит памяти Hexagon NPU

> **Важно об архитектуре 32-битного Hexagon cDSP (`v73`–`v79`)**:
> Микропроцессор Hexagon HTP внутри чипсетов Snapdragon 8s Gen 3 (`v73`), 8 Gen 3 (`v75`) и 8 Elite (`v79`) использует **32-битное виртуальное адресное пространство FastRPC (~`3.0–3.2 ГБ` на одну сессию `HTP0`)**, где одновременно размещаются переупакованные веса модели, KV-кэш и рабочие графы. Поэтому даже на смартфонах с 12–16 ГБ ОЗУ максимальный размер весов для строгого запуска целиком на одном `HTP0` составляет **до ~`2.4–2.5 ГБ`** (модели `9B` весом ~5.3 ГБ автоматически отклоняются проверкой безопасности с подробным пояснением).

Наилучшую точность и скорость на бэкенде `ggml-hexagon` обеспечивают форматы квантования **`Q4_0`** и **`Q8_0`**:
- **Сверхбыстрые компактные модели (`0.5B–1.7B`)**:
  - `Qwen3-0.6B-Q4_0.gguf` / `Qwen3-0.6B-Q8_0.gguf`
  - `Qwen3-1.7B-Q4_0.gguf`
  - `Llama-3.2-1B-Instruct-Q4_0.gguf` / `Q8_0.gguf`
- **Оптимальный баланс качества и памяти для `HTP v73` (`2B–3B`, вес `1.3–2.0 ГБ`)**:
  - `Qwen2.5-3B-Instruct-Q4_0.gguf`
  - `Llama-3.2-3B-Instruct-Q4_0.gguf`
  - `Gemma-2-2B-IT-Q4_0.gguf` / `Phi-3.5-mini-instruct-Q4_0.gguf` (до ~2.3 ГБ с включённым `Q8_0 KV-кэшем` и контекстом `2048–4096`).

---

## Сборка через Gradle

### Требования для сборки
- **JDK 17**
- **Android SDK** (`compileSdk = 36`, **NDK `28.2.13676358`**, **CMake `3.31.6`**)

### Быстрая сборка APK (автозагрузка официального пакета `llama.cpp` Snapdragon Hexagon NPU)
Задача Gradle `prepareSnapdragonNpuSdk` автоматически скачивает и проверяет официальный архив `llama-b12211-bin-android-arm64-snapdragon.tar.gz` (включающий `libggml-hexagon.so` и DSP-модули `libggml-htp-v73.so` .. `v85.so`):

```bash
# Запуск модульных тестов и проверок Lint
./gradlew testDebugUnitTest lintDebug

# Сборка Debug APK
./gradlew assembleDebug

# Сборка подписанного Release APK v0.0.2
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
│   ├── build.gradle.kts                  # Конфигурация Gradle (v0.0.2) + задача prepareSnapdragonNpuSdk
│   ├── proguard-rules.pro                # Правила R8 для сохранения JNI-моста в Release
│   └── src/
│       ├── main/
│       │   ├── AndroidManifest.xml       # Доступ к libcdsprpc.so, extractNativeLibs=true, allowBackup=false
│       │   ├── cpp/
│       │   │   ├── CMakeLists.txt        # Сборка libnpu_llama_jni.so и проверка наличия libggml-htp-v73.so
│       │   │   ├── npu_engine.hpp        # Интерфейс C++ движка Hexagon NPU и инспектора GGUF
│       │   │   ├── npu_engine.cpp        # HVX/HMX политика, GGUF preflight, Qwen3 <think>, защита OOM
│       │   │   └── npu_llama_jni.cpp     # JNI-мост с перехватом C++ исключений и UTF-16 <-> UTF-8
│       │   ├── java/ru/npu/llamachat/
│       │   │   ├── MainActivity.kt       # Экран чата, SAF-пикер .gguf, диалоги ошибок, панель токенов и ОЗУ
│       │   │   ├── engine/               # SnapdragonNpuDetector, NpuMemoryBudgetEvaluator, NpuLlamaBridge, NpuChatEngine
│       │   │   ├── model/                # ModelConfig (вкл. <think>), ChatMessage, NpuTelemetry
│       │   │   └── ui/                   # ChatViewModel, ChatAdapter, SettingsBottomSheet
│       │   └── res/                      # Material 3 XML-разметка и ресурсы
│       └── test/                         # Модульные JUnit-тесты детектора SoC, бюджета памяти NPU и конфига
├── scripts/
│   └── build-hexagon-sdk.sh              # Скрипт сборки Hexagon SDK из исходников в Docker
├── .github/workflows/
│   └── android-release.yml               # CI/CD сборка, тесты, Lint и публикация Release APK v0.0.2
├── CHANGELOG.md
└── README.md
```

### Локальный API (экспериментально)

Во вкладке «API» включите сервис и разрешите уведомления. Сервис использует общую с чатом модель и показывает уведомление с действиями выбора/выгрузки модели и остановки API. Выбор модели из уведомления открывает системный выбор файла. Уведомление и foreground service повышают приоритет процесса, но не гарантируют его сохранение при давлении на память или принудительной остановке.

API доступен только с того же устройства на `http://127.0.0.1:8765`; новый случайный Bearer token отображается во вкладке при каждом запуске. Не передавайте его другим приложениям или сайтам. Поддерживаются `GET /v1/models` и непотоковый `POST /v1/chat/completions`: поля `model` (необязательно, имя загруженного файла), `messages` (роли system, user, assistant) и `stream: false`. Остальные поля отклоняются; параметры генерации задаются в настройках приложения. Одновременно обслуживается один запрос, история запроса не сохраняется на диск. Это не полный OpenAI API; доступ из локальной сети не предусмотрен.

Генератор изображений на QNN/QAIRT пока не реализован и на SM8635 не проверен: см. `docs/issledovanie-npu-izobrazheniya.md`.

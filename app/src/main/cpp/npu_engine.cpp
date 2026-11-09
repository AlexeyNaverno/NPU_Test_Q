#include "npu_engine.hpp"

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>

#if defined(__ANDROID__)
#include <android/log.h>
#define NPU_LOG_TAG "NpuLlamaEngine"
#define NPU_LOGI(...) __android_log_print(ANDROID_LOG_INFO,  NPU_LOG_TAG, __VA_ARGS__)
#define NPU_LOGW(...) __android_log_print(ANDROID_LOG_WARN,  NPU_LOG_TAG, __VA_ARGS__)
#define NPU_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, NPU_LOG_TAG, __VA_ARGS__)
#else
#define NPU_LOGI(...) do { std::fprintf(stderr, "[INFO] " __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
#define NPU_LOGW(...) do { std::fprintf(stderr, "[WARN] " __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
#define NPU_LOGE(...) do { std::fprintf(stderr, "[ERR ] " __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
#endif

namespace npu {

namespace {

static NpuEngine * g_active_log_target = nullptr;

void llama_log_trampoline(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (!text) {
        return;
    }
#if defined(__ANDROID__)
    int prio = ANDROID_LOG_DEBUG;
    if (level == GGML_LOG_LEVEL_INFO)  prio = ANDROID_LOG_INFO;
    if (level == GGML_LOG_LEVEL_WARN)  prio = ANDROID_LOG_WARN;
    if (level == GGML_LOG_LEVEL_ERROR) prio = ANDROID_LOG_ERROR;
    __android_log_write(prio, "llama-npu", text);
#endif
    if (g_active_log_target) {
        g_active_log_target->onNativeLogLine(static_cast<int>(level), text);
    }
}

struct ProgressCallbackBridge {
    const std::function<void(float)> * fn = nullptr;
    std::atomic<bool> * cancel_flag       = nullptr;
};

bool model_load_progress_cb(float progress, void * user_data) {
    auto * bridge = static_cast<ProgressCallbackBridge *>(user_data);
    if (bridge) {
        if (bridge->cancel_flag && bridge->cancel_flag->load(std::memory_order_relaxed)) {
            return false;
        }
        if (bridge->fn && *(bridge->fn)) {
            (*(bridge->fn))(progress);
        }
    }
    return true;
}

bool decode_abort_cb(void * data) {
    auto * flag = static_cast<std::atomic<bool> *>(data);
    return flag && flag->load(std::memory_order_relaxed);
}

struct ScopedBatch {
    llama_batch batch{};
    explicit ScopedBatch(int32_t max_tokens) {
        batch = llama_batch_init(max_tokens, 0, 1);
    }
    ~ScopedBatch() {
        llama_batch_free(batch);
    }
    ScopedBatch(const ScopedBatch &) = delete;
    ScopedBatch & operator=(const ScopedBatch &) = delete;
};

bool is_valid_power_mode(const std::string & mode) {
    return mode == "low_power_saver" ||
           mode == "power_saver" ||
           mode == "high_power_saver" ||
           mode == "low_balanced" ||
           mode == "balanced" ||
           mode == "high_performance" ||
           mode == "sustained_high_performance" ||
           mode == "burst" ||
           mode == "default";
}

size_t utf8_expected_len(unsigned char lead) {
    if ((lead & 0x80u) == 0x00u) return 1;
    if ((lead & 0xE0u) == 0xC0u) return 2;
    if ((lead & 0xF0u) == 0xE0u) return 3;
    if ((lead & 0xF8u) == 0xF0u) return 4;
    return 0; // Invalid lead byte or stray continuation byte
}

std::string format_chat_messages(
    const llama_model * model,
    const std::vector<ChatTurnMessage> & messages) {
    std::vector<llama_chat_message> c_msgs;
    c_msgs.reserve(messages.size());
    size_t approx_chars = 64;
    for (const auto & m : messages) {
        c_msgs.push_back(llama_chat_message{m.role.c_str(), m.content.c_str()});
        approx_chars += m.role.size() + m.content.size() + 32;
    }

    const char * model_tmpl = model ? llama_model_chat_template(model, nullptr) : nullptr;
    std::vector<char> buf(approx_chars * 2 + 256, '\0');

    auto try_apply = [&](const char * tmpl) -> int32_t {
        int32_t res = llama_chat_apply_template(
            tmpl,
            c_msgs.data(),
            c_msgs.size(),
            /*add_ass=*/true,
            buf.data(),
            static_cast<int32_t>(buf.size()));
        if (res > static_cast<int32_t>(buf.size())) {
            buf.assign(static_cast<size_t>(res) + 64, '\0');
            res = llama_chat_apply_template(
                tmpl,
                c_msgs.data(),
                c_msgs.size(),
                /*add_ass=*/true,
                buf.data(),
                static_cast<int32_t>(buf.size()));
        }
        return res;
    };

    int32_t written = try_apply(model_tmpl);
    if (written < 0) {
        // Fallback to standard ChatML if GGUF embeds an unrecognized custom Jinja macro
        written = try_apply("chatml");
    }

    if (written >= 0 && static_cast<size_t>(written) <= buf.size()) {
        return std::string(buf.data(), static_cast<size_t>(written));
    }

    // Ultimate deterministic fallback
    std::ostringstream oss;
    for (const auto & m : messages) {
        oss << "<|im_start|>" << m.role << "\n" << m.content << "<|im_end|>\n";
    }
    oss << "<|im_start|>assistant\n";
    return oss.str();
}

std::vector<llama_token> tokenize_string(
    const llama_vocab * vocab,
    const std::string & text,
    bool add_special,
    bool parse_special) {
    if (!vocab || text.empty()) {
        return {};
    }
    int32_t n_est = static_cast<int32_t>(text.size()) + 16;
    std::vector<llama_token> tokens(static_cast<size_t>(n_est));
    int32_t n = llama_tokenize(
        vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        add_special,
        parse_special);
    if (n < 0 && n != INT32_MIN) {
        tokens.resize(static_cast<size_t>(-n));
        n = llama_tokenize(
            vocab,
            text.data(),
            static_cast<int32_t>(text.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            add_special,
            parse_special);
    }
    if (n < 0) {
        return {};
    }
    tokens.resize(static_cast<size_t>(n));
    return tokens;
}

} // namespace

// ---------------------------------------------------------------------------
// EngineConfig
// ---------------------------------------------------------------------------

void EngineConfig::sanitize() {
    n_ctx          = std::clamp(n_ctx, 256, 32768);
    max_new_tokens = std::clamp(max_new_tokens, 16, 8192);
    n_batch        = std::clamp(n_batch, 32, 2048);
    if (n_batch > n_ctx) {
        n_batch = n_ctx;
    }
    n_threads      = std::clamp(n_threads, 1, 8);
    // STRICT NPU POLICY: Never allow 0 offloaded layers! <= 0 or > 999 maps to 999 (all layers on Hexagon HTP)
    if (n_npu_layers <= 0 || n_npu_layers > 999) {
        n_npu_layers = 999;
    }
    temperature    = std::clamp(temperature, 0.0f, 2.0f);
    top_p          = std::clamp(top_p, 0.05f, 1.0f);
    top_k          = std::clamp(top_k, 1, 500);
    min_p          = std::clamp(min_p, 0.0f, 0.95f);
    repeat_penalty = std::clamp(repeat_penalty, 1.0f, 2.5f);
    repeat_last_n  = std::clamp(repeat_last_n, 0, 512);
    flash_attn_mode = std::clamp(flash_attn_mode, -1, 1);

    if (!is_valid_power_mode(power_mode)) {
        power_mode = "sustained_high_performance";
    }
    if (htp_devices.empty()) {
        htp_devices = "HTP0";
    }
}

// ---------------------------------------------------------------------------
// Utf8StreamAssembler
// ---------------------------------------------------------------------------

std::string Utf8StreamAssembler::push(const char * bytes, size_t len) {
    if (!bytes || len == 0) {
        return {};
    }
    pending_.append(bytes, len);

    size_t valid_end = 0;
    const size_t total = pending_.size();
    size_t i = 0;

    while (i < total) {
        const auto lead = static_cast<unsigned char>(pending_[i]);
        const size_t seq_len = utf8_expected_len(lead);
        if (seq_len == 0) {
            // Replace invalid stray byte with UTF-8 replacement char U+FFFD and advance
            pending_.replace(i, 1, "\xEF\xBF\xBD");
            return push("", 0);
        }
        if (i + seq_len > total) {
            // Trailing multi-byte character is incomplete — wait for next token!
            break;
        }
        bool cont_ok = true;
        for (size_t k = 1; k < seq_len; ++k) {
            const auto c = static_cast<unsigned char>(pending_[i + k]);
            if ((c & 0xC0u) != 0x80u) {
                cont_ok = false;
                break;
            }
        }
        if (!cont_ok) {
            pending_.replace(i, 1, "\xEF\xBF\xBD");
            return push("", 0);
        }
        i += seq_len;
        valid_end = i;
    }

    if (valid_end == 0) {
        return {};
    }
    std::string ready = pending_.substr(0, valid_end);
    pending_.erase(0, valid_end);
    return ready;
}

std::string Utf8StreamAssembler::flush() {
    if (pending_.empty()) {
        return {};
    }
    pending_.clear();
    return "\xEF\xBF\xBD";
}

void Utf8StreamAssembler::clear() {
    pending_.clear();
}

// ---------------------------------------------------------------------------
// NpuEngine
// ---------------------------------------------------------------------------

NpuEngine::NpuEngine() {
    g_active_log_target = this;
}

NpuEngine::~NpuEngine() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    unloadModelLocked();
    if (g_active_log_target == this) {
        llama_log_set(nullptr, nullptr);
        g_active_log_target = nullptr;
    }
}

bool NpuEngine::validateGgufHeaderBytes(const uint8_t * data, size_t len, std::string * err_out) {
    if (!data || len < 8) {
        if (err_out) *err_out = "Файл слишком мал и не содержит заголовок GGUF.";
        return false;
    }
    if (!(data[0] == 'G' && data[1] == 'G' && data[2] == 'U' && data[3] == 'F')) {
        if (err_out) *err_out = "Неверная сигнатура файла: ожидается формат .gguf (заголовок GGUF).";
        return false;
    }
    uint32_t version = static_cast<uint32_t>(data[4]) |
                       (static_cast<uint32_t>(data[5]) << 8) |
                       (static_cast<uint32_t>(data[6]) << 16) |
                       (static_cast<uint32_t>(data[7]) << 24);
    if (version < 2 || version > 3) {
        if (err_out) {
            *err_out = "Неподдерживаемая версия формата GGUF v" + std::to_string(version) +
                       " (поддерживаются GGUF v2 и v3).";
        }
        return false;
    }
    return true;
}

void NpuEngine::onNativeLogLine(int level, const char * text) {
    if (!text) return;
    std::lock_guard<std::mutex> lk(log_parse_mutex_);
    log_line_accum_.append(text);

    size_t nl = 0;
    while ((nl = log_line_accum_.find('\n')) != std::string::npos) {
        std::string line = log_line_accum_.substr(0, nl);
        log_line_accum_.erase(0, nl + 1);

        // 1. Parse Hexagon architecture version (e.g. "ggml-hex: Hexagon Arch version v73")
        {
            const char * key = "Hexagon Arch version v";
            auto pos = line.find(key);
            if (pos != std::string::npos) {
                parsed_hexagon_arch_ = std::atoi(line.c_str() + pos + std::strlen(key));
            }
        }

        // 2. Parse Hexagon hwinfo ("hwinfo: threads 6, hvx 4, hmx 1, vtcm 8 MB")
        {
            const char * key = "hwinfo: threads ";
            auto pos = line.find(key);
            if (pos != std::string::npos) {
                int th = 0, hvx = 0, hmx = 0, vtcm = 0;
                if (std::sscanf(line.c_str() + pos,
                                "hwinfo: threads %d, hvx %d, hmx %d, vtcm %d MB",
                                &th, &hvx, &hmx, &vtcm) == 4) {
                    parsed_hvx_     = hvx;
                    parsed_hmx_     = hmx;
                    parsed_vtcm_mb_ = vtcm;
                }
            }
        }

        // 3. Parse offloaded layers ("offloaded 17/17 layers to GPU")
        {
            const char * key = "offloaded ";
            auto pos = line.find(key);
            if (pos != std::string::npos && line.find("layers to GPU") != std::string::npos) {
                int npu_l = 0, tot_l = 0;
                if (std::sscanf(line.c_str() + pos, "offloaded %d/%d layers", &npu_l, &tot_l) == 2) {
                    parsed_layers_npu_   = npu_l;
                    parsed_layers_total_ = tot_l;
                }
            }
        }

        // 4. Parse buffer sizes ("HTP0 model buffer size =   504.26 MiB")
        {
            const char * key = "model buffer size =";
            auto pos = line.find(key);
            if (pos != std::string::npos) {
                double mib = std::atof(line.c_str() + pos + std::strlen(key));
                if (line.find("HTP") != std::string::npos && line.find("HOST") == std::string::npos) {
                    parsed_htp_mib_ += mib;
                } else if (line.find("CPU") != std::string::npos || line.find("HOST") != std::string::npos) {
                    parsed_cpu_mib_ += mib;
                }
            }
        }

        // 5. Capture Hexagon FastRPC error lines for diagnostics
        if (level >= static_cast<int>(GGML_LOG_LEVEL_ERROR) && line.find("ggml-hex:") != std::string::npos) {
            last_hex_error_ = line;
        }
    }
}

InitResult NpuEngine::init(
    const std::string & native_lib_dir,
    const std::string & htp_devices,
    const std::string & power_mode) {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);

    InitResult res{};
    native_lib_dir_ = native_lib_dir;

    const std::string clean_devices = htp_devices.empty() ? "HTP0" : htp_devices;
    const std::string clean_power   = is_valid_power_mode(power_mode) ? power_mode : "sustained_high_performance";

    if (!native_lib_dir.empty()) {
        std::error_code ec;
        res.has_v73_skel = std::filesystem::exists(
            std::filesystem::path(native_lib_dir) / "libggml-htp-v73.so", ec);

        // Configure Qualcomm FastRPC & GGML Hexagon NPU environment BEFORE backend registration!
        ::setenv("GGML_HEXAGON_LIBHTP_DIR", native_lib_dir.c_str(), 1);
        ::setenv("ADSP_LIBRARY_PATH",       native_lib_dir.c_str(), 1);
        ::setenv("CDSP_LIBRARY_PATH",       native_lib_dir.c_str(), 1);
    }

    ::setenv("GGML_HEXAGON_DEVICES",    clean_devices.c_str(), 1);
    ::setenv("GGML_HEXAGON_POWER_MODE", clean_power.c_str(),   1);
    ::setenv("GGML_HEXAGON_OPPOLL",     "1",                   1);
    ::setenv("GGML_HEXAGON_HOSTBUF",    "1",                   1);

    // Defense-in-depth: block all ops on OpenCL Adreno GPU even if libggml-opencl.so is in memory
    ::setenv("GGML_OPENCL_OPFILTER",    ".*",                  1);

    llama_log_set(llama_log_trampoline, nullptr);

    try {
        if (!backend_initialized_.load()) {
            if (!native_lib_dir.empty()) {
                // Load dynamic backends if built with GGML_BACKEND_DL=ON
                std::error_code ec;
                const auto hex_so = std::filesystem::path(native_lib_dir) / "libggml-hexagon.so";
                if (std::filesystem::exists(hex_so, ec)) {
                    ggml_backend_load(hex_so.c_str());
                }
            }
            llama_backend_init();
            backend_initialized_.store(true);
        }

        // Enumerate ONLY Qualcomm Hexagon HTP NPU devices — strictly exclude CPU, OpenCL GPU, Vulkan, etc.
        htp_devices_.clear();
        htp_devices_null_term_.clear();
        std::vector<std::string> labels;

        const size_t dev_count = ggml_backend_dev_count();
        for (size_t i = 0; i < dev_count; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!dev) continue;

            const char * d_name = ggml_backend_dev_name(dev);
            const char * d_desc = ggml_backend_dev_description(dev);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            const char * r_name = reg ? ggml_backend_reg_name(reg) : "";

            const std::string dev_name = d_name ? d_name : "";
            const std::string dev_desc = d_desc ? d_desc : "";
            const std::string reg_name = r_name ? r_name : "";

            const bool is_hexagon_htp =
                (reg_name == "HTP" || dev_desc == "Hexagon") &&
                dev_name.rfind("HTP", 0) == 0;

            if (is_hexagon_htp) {
                htp_devices_.push_back(dev);
                labels.push_back(dev_name + " (" + dev_desc + ")");
            } else {
                NPU_LOGI("Ignoring non-NPU device '%s' (%s, reg=%s) per strict NPU-only policy",
                         dev_name.c_str(), dev_desc.c_str(), reg_name.c_str());
            }
        }

        for (auto * d : htp_devices_) {
            htp_devices_null_term_.push_back(d);
        }
        htp_devices_null_term_.push_back(nullptr);

    } catch (const std::exception & e) {
        npu_ready_.store(false);
        res.ok = false;
        res.error_message =
            std::string("Ошибка инициализации сессии Qualcomm Hexagon NPU (cDSP FastRPC): ") +
            e.what() +
            ". Выполнение на CPU/GPU заблокировано политикой приложения.";
        NPU_LOGE("%s", res.error_message.c_str());
        return res;
    }

    int32_t arch = 0, hvx = 0, hmx = 0, vtcm = 0;
    std::string hex_err;
    {
        std::lock_guard<std::mutex> lk(log_parse_mutex_);
        arch    = parsed_hexagon_arch_;
        hvx     = parsed_hvx_;
        hmx     = parsed_hmx_;
        vtcm    = parsed_vtcm_mb_;
        hex_err = last_hex_error_;
    }

    res.hexagon_arch   = arch;
    res.htp_device_cnt = static_cast<int32_t>(htp_devices_.size());

    if (htp_devices_.empty()) {
        npu_ready_.store(false);
        res.ok = false;
        std::ostringstream err;
        err << "Ускоритель Qualcomm Hexagon NPU (HTP0) не обнаружен или недоступен. "
            << "Приложение работает ИСКЛЮЧИТЕЛЬНО на NPU Snapdragon (8s Gen 3 HTP v73 и совместимых) — "
            << "запуск на CPU или GPU строго запрещён.";
        if (!hex_err.empty()) {
            err << "\nДетали драйвера: " << hex_err;
        }
        res.error_message = err.str();
        NPU_LOGE("%s", res.error_message.c_str());
        return res;
    }

    std::ostringstream dev_oss;
    for (size_t i = 0; i < htp_devices_.size(); ++i) {
        if (i > 0) dev_oss << ", ";
        dev_oss << ggml_backend_dev_name(htp_devices_[i]);
    }
    active_devices_label_ = dev_oss.str();
    res.active_devices    = active_devices_label_;

    std::ostringstream hw;
    hw << "Hexagon HTP";
    if (arch > 0) hw << " v" << arch;
    hw << " (" << active_devices_label_ << ")";
    if (hvx > 0 || hmx > 0 || vtcm > 0) {
        hw << " • HVX: " << hvx << " • HMX: " << hmx << " • VTCM: " << vtcm << " МБ";
    }
    res.hw_summary = hw.str();

    npu_ready_.store(true);
    res.ok = true;
    NPU_LOGI("Hexagon NPU initialized successfully: %s", res.hw_summary.c_str());
    return res;
}

ModelLoadResult NpuEngine::loadModelFromFd(
    int detached_fd,
    const std::string & /*display_name*/,
    const EngineConfig & config,
    const std::function<void(float)> & on_progress) {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);

    ModelLoadResult out{};
    if (detached_fd < 0) {
        out.error_message = "Некорректный файловый дескриптор выбранного файла .gguf.";
        return out;
    }

    FILE * fp = ::fdopen(detached_fd, "rb");
    if (!fp) {
        ::close(detached_fd);
        out.error_message = "Не удалось открыть поток чтения .gguf файла.";
        return out;
    }

    struct FileCloser {
        FILE * f;
        ~FileCloser() {
            if (f) std::fclose(f);
        }
    } closer{fp};

    // HARD GATE 1: Verify Hexagon NPU readiness BEFORE touching llama_model_load
    if (!npu_ready_.load() || htp_devices_.empty()) {
        out.error_message =
            "Загрузка модели отклонена: Qualcomm Hexagon NPU (HTP0) не активен. "
            "Запуск модели на CPU или GPU строго заблокирован.";
        return out;
    }

    // Validate 8-byte GGUF magic header
    uint8_t header[8] = {};
    if (std::fread(header, 1, sizeof(header), fp) != sizeof(header)) {
        out.error_message = "Не удалось прочитать заголовок файла .gguf.";
        return out;
    }
    if (!validateGgufHeaderBytes(header, sizeof(header), &out.error_message)) {
        return out;
    }
    if (std::fseek(fp, 0, SEEK_SET) != 0) {
        out.error_message = "Выбранный источник файла не поддерживает произвольное позиционирование (seek).";
        return out;
    }

    unloadModelLocked();
    cancel_requested_.store(false);

    EngineConfig cfg = config;
    cfg.sanitize();
    ::setenv("GGML_HEXAGON_POWER_MODE", cfg.power_mode.c_str(), 1);

    {
        std::lock_guard<std::mutex> lk(log_parse_mutex_);
        parsed_layers_npu_   = 0;
        parsed_layers_total_ = 0;
        parsed_htp_mib_      = 0.0;
        parsed_cpu_mib_      = 0.0;
        last_hex_error_.clear();
    }

    llama_model_params mparams = llama_model_default_params();
    // STRICT NPU PINNING: Pass ONLY Hexagon HTP devices!
    mparams.devices      = htp_devices_null_term_.data();
    mparams.n_gpu_layers = cfg.n_npu_layers; // Always >= 1 (default 999 = all layers on Hexagon NPU)
    mparams.load_mode    = LLAMA_LOAD_MODE_NONE; // --no-mmap: required for 2D HMX/HVX tile repacking in RPCMEM
    mparams.split_mode   = (htp_devices_.size() > 1) ? LLAMA_SPLIT_MODE_LAYER : LLAMA_SPLIT_MODE_NONE;

    ProgressCallbackBridge prog_bridge{&on_progress, &cancel_requested_};
    mparams.progress_callback           = model_load_progress_cb;
    mparams.progress_callback_user_data = &prog_bridge;

    llama_model * loaded = nullptr;
    try {
        loaded = llama_model_load_from_file_ptr(fp, mparams);
    } catch (const std::exception & e) {
        out.error_message = std::string("Исключение при загрузке GGUF на Hexagon NPU: ") + e.what();
        return out;
    }

    return finishModelLoadLocked(loaded, cfg);
}

ModelLoadResult NpuEngine::loadModelFromPath(
    const std::string & file_path,
    const EngineConfig & config,
    const std::function<void(float)> & on_progress) {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);

    ModelLoadResult out{};
    if (!npu_ready_.load() || htp_devices_.empty()) {
        out.error_message =
            "Загрузка модели отклонена: Qualcomm Hexagon NPU (HTP0) не активен. "
            "Запуск модели на CPU или GPU строго заблокирован.";
        return out;
    }

    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs) {
        out.error_message = "Не удалось открыть файл модели: " + file_path;
        return out;
    }
    uint8_t header[8] = {};
    ifs.read(reinterpret_cast<char *>(header), sizeof(header));
    if (ifs.gcount() != static_cast<std::streamsize>(sizeof(header))) {
        out.error_message = "Файл повреждён или пуст.";
        return out;
    }
    if (!validateGgufHeaderBytes(header, sizeof(header), &out.error_message)) {
        return out;
    }
    ifs.close();

    unloadModelLocked();
    cancel_requested_.store(false);

    EngineConfig cfg = config;
    cfg.sanitize();
    ::setenv("GGML_HEXAGON_POWER_MODE", cfg.power_mode.c_str(), 1);

    {
        std::lock_guard<std::mutex> lk(log_parse_mutex_);
        parsed_layers_npu_   = 0;
        parsed_layers_total_ = 0;
        parsed_htp_mib_      = 0.0;
        parsed_cpu_mib_      = 0.0;
        last_hex_error_.clear();
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.devices      = htp_devices_null_term_.data();
    mparams.n_gpu_layers = cfg.n_npu_layers;
    mparams.load_mode    = LLAMA_LOAD_MODE_NONE;
    mparams.split_mode   = (htp_devices_.size() > 1) ? LLAMA_SPLIT_MODE_LAYER : LLAMA_SPLIT_MODE_NONE;

    ProgressCallbackBridge prog_bridge{&on_progress, &cancel_requested_};
    mparams.progress_callback           = model_load_progress_cb;
    mparams.progress_callback_user_data = &prog_bridge;

    llama_model * loaded = nullptr;
    try {
        loaded = llama_model_load_from_file(file_path.c_str(), mparams);
    } catch (const std::exception & e) {
        out.error_message = std::string("Исключение при загрузке GGUF на Hexagon NPU: ") + e.what();
        return out;
    }

    return finishModelLoadLocked(loaded, cfg);
}

ModelLoadResult NpuEngine::finishModelLoadLocked(
    llama_model * loaded_model,
    const EngineConfig & sanitized_cfg) {
    ModelLoadResult out{};

    if (!loaded_model) {
        if (cancel_requested_.load()) {
            out.error_message = "Загрузка модели отменена пользователем.";
        } else {
            std::lock_guard<std::mutex> lk(log_parse_mutex_);
            out.error_message =
                "Не удалось загрузить GGUF модель в память Qualcomm Hexagon NPU. "
                "Проверьте целостность файла и достаточность свободного ОЗУ.";
            if (!last_hex_error_.empty()) {
                out.error_message += "\n" + last_hex_error_;
            }
        }
        return out;
    }

    int32_t log_npu_layers = 0;
    int32_t log_tot_layers = 0;
    double  htp_mib        = 0.0;
    double  cpu_mib        = 0.0;
    {
        std::lock_guard<std::mutex> lk(log_parse_mutex_);
        log_npu_layers = parsed_layers_npu_;
        log_tot_layers = parsed_layers_total_;
        htp_mib        = parsed_htp_mib_;
        cpu_mib        = parsed_cpu_mib_;
    }

    const int32_t model_layers = llama_model_n_layer(loaded_model);

    // HARD GATE 2: Verify weights were actually placed onto Hexagon HTP buffers!
    if (log_npu_layers <= 0 && htp_mib <= 0.01) {
        llama_model_free(loaded_model);
        out.error_message =
            "ОШИБКА NPU-ВЕРИФИКАЦИИ: Ни один слой модели не был размещён в буферах Hexagon HTP0! "
            "Запуск чисто на CPU заблокирован. Используйте модель с поддерживаемым NPU квантованием "
            "(рекомендуется Q4_0, Q8_0, Q4_K_M, IQ4_NL или F16).";
        return out;
    }

    {
        std::unique_lock<std::shared_mutex> vlock(vocab_mutex_);
        model_ = loaded_model;
        vocab_ = llama_model_get_vocab(model_);
    }

    active_config_ = sanitized_cfg;

    std::string ctx_err;
    if (!recreateContextLocked(active_config_, &ctx_err)) {
        unloadModelLocked();
        out.error_message = ctx_err;
        return out;
    }

    char desc_buf[256] = {};
    llama_model_desc(model_, desc_buf, sizeof(desc_buf));
    const enum llama_ftype ftype = llama_model_ftype(model_);
    const char * ftype_str = llama_ftype_name(ftype);

    npu_htp_buffer_mib_.store(htp_mib);
    host_weight_mib_.store(cpu_mib);
    model_loaded_.store(true);

    out.ok              = true;
    out.model_desc      = desc_buf[0] ? desc_buf : "GGUF LLM";
    out.quant_name      = ftype_str ? ftype_str : "GGUF";
    out.model_bytes     = llama_model_size(model_);
    out.model_params    = llama_model_n_params(model_);
    out.n_ctx_train     = llama_model_n_ctx_train(model_);
    out.n_ctx_active    = static_cast<int32_t>(llama_n_ctx(ctx_));
    out.n_layers_total  = (log_tot_layers > 0) ? log_tot_layers : model_layers;
    out.n_layers_npu    = (log_npu_layers > 0) ? log_npu_layers : model_layers;
    out.htp_buffer_mib  = htp_mib;
    out.host_buffer_mib = cpu_mib;
    out.npu_device_used = active_devices_label_;

    NPU_LOGI("Model loaded on Hexagon NPU (%s): %s [%s], NPU layers %d/%d, HTP buffer %.2f MiB, n_ctx=%d",
             out.npu_device_used.c_str(),
             out.model_desc.c_str(),
             out.quant_name.c_str(),
             out.n_layers_npu,
             out.n_layers_total,
             out.htp_buffer_mib,
             out.n_ctx_active);

    return out;
}

bool NpuEngine::recreateContextLocked(const EngineConfig & cfg, std::string * err_out) {
    if (!model_) {
        if (err_out) *err_out = "Модель не загружена.";
        return false;
    }

    if (sampler_) {
        llama_sampler_free(sampler_);
        sampler_ = nullptr;
    }
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    session_tokens_.clear();
    context_used_tokens_.store(0);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = static_cast<uint32_t>(cfg.n_ctx);
    cparams.n_batch         = static_cast<uint32_t>(cfg.n_batch);
    cparams.n_ubatch        = static_cast<uint32_t>(cfg.n_batch);
    cparams.n_threads       = cfg.n_threads;
    cparams.n_threads_batch = cfg.n_threads;
    cparams.offload_kqv     = true; // Keep KV-cache & attention KQV on Hexagon NPU!
    cparams.op_offload      = true; // Offload host tensor ops to Hexagon NPU!
    cparams.no_perf         = false;

    if (cfg.flash_attn_mode > 0) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    } else if (cfg.flash_attn_mode == 0) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    } else {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    }

    const enum ggml_type kv_type = cfg.kv_q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    cparams.type_k = kv_type;
    cparams.type_v = kv_type;

    cparams.abort_callback      = decode_abort_cb;
    cparams.abort_callback_data = &cancel_requested_;

    try {
        ctx_ = llama_init_from_model(model_, cparams);
    } catch (const std::exception & e) {
        if (err_out) {
            *err_out = std::string("Ошибка выделения контекста на Hexagon NPU: ") + e.what();
        }
        return false;
    }

    if (!ctx_) {
        if (err_out) {
            *err_out = "Не удалось выделить KV-кэш контекста (" + std::to_string(cfg.n_ctx) +
                       " токенов) на Hexagon NPU. Попробуйте уменьшить макс. контекст или включить Q8_0 KV-кэш.";
        }
        return false;
    }

    context_max_tokens_.store(static_cast<int32_t>(llama_n_ctx(ctx_)));
    rebuildSamplerLocked(cfg);
    return true;
}

void NpuEngine::rebuildSamplerLocked(const EngineConfig & cfg) {
    if (sampler_) {
        llama_sampler_free(sampler_);
        sampler_ = nullptr;
    }
    if (!model_) {
        return;
    }

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    sampler_ = llama_sampler_chain_init(sparams);

    if (cfg.repeat_penalty > 1.001f && cfg.repeat_last_n > 0 && vocab_) {
        llama_sampler_chain_add(
            sampler_,
            llama_sampler_init_penalties(
                llama_vocab_n_tokens(vocab_),
                cfg.repeat_last_n,
                cfg.repeat_penalty,
                0.0f,
                0.0f));
    }

    if (cfg.temperature <= 0.0001f) {
        llama_sampler_chain_add(sampler_, llama_sampler_init_greedy());
    } else {
        if (cfg.top_k > 0) {
            llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(cfg.top_k));
        }
        if (cfg.top_p < 0.9999f) {
            llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(cfg.top_p, 1));
        }
        if (cfg.min_p > 0.0001f) {
            llama_sampler_chain_add(sampler_, llama_sampler_init_min_p(cfg.min_p, 1));
        }
        llama_sampler_chain_add(sampler_, llama_sampler_init_temp(cfg.temperature));
        const uint32_t seed = (cfg.seed < 0)
            ? LLAMA_DEFAULT_SEED
            : static_cast<uint32_t>(cfg.seed);
        llama_sampler_chain_add(sampler_, llama_sampler_init_dist(seed));
    }
}

bool NpuEngine::applyConfig(const EngineConfig & config, std::string * err_out) {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);

    EngineConfig cfg = config;
    cfg.sanitize();
    ::setenv("GGML_HEXAGON_POWER_MODE", cfg.power_mode.c_str(), 1);

    if (!model_ || !ctx_) {
        active_config_ = cfg;
        return true;
    }

    const bool need_ctx_recreate =
        cfg.n_ctx != active_config_.n_ctx ||
        cfg.n_batch != active_config_.n_batch ||
        cfg.flash_attn_mode != active_config_.flash_attn_mode ||
        cfg.kv_q8 != active_config_.kv_q8;

    if (need_ctx_recreate) {
        if (!recreateContextLocked(cfg, err_out)) {
            return false;
        }
    } else {
        if (cfg.n_threads != active_config_.n_threads) {
            llama_set_n_threads(ctx_, cfg.n_threads, cfg.n_threads);
        }
        rebuildSamplerLocked(cfg);
    }

    active_config_ = cfg;
    return true;
}

void NpuEngine::clearChat() {
    cancel_requested_.store(true);
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    session_tokens_.clear();
    if (ctx_) {
        llama_memory_t mem = llama_get_memory(ctx_);
        if (mem) {
            llama_memory_clear(mem, true);
        }
    }
    if (sampler_) {
        llama_sampler_reset(sampler_);
    }
    context_used_tokens_.store(0);
    cancel_requested_.store(false);
}

void NpuEngine::unloadModel() {
    cancel_requested_.store(true);
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    unloadModelLocked();
    cancel_requested_.store(false);
}

void NpuEngine::unloadModelLocked() {
    model_loaded_.store(false);
    session_tokens_.clear();
    context_used_tokens_.store(0);
    context_max_tokens_.store(0);
    npu_htp_buffer_mib_.store(0.0);
    host_weight_mib_.store(0.0);

    if (sampler_) {
        llama_sampler_free(sampler_);
        sampler_ = nullptr;
    }
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    {
        std::unique_lock<std::shared_mutex> vlock(vocab_mutex_);
        vocab_ = nullptr;
        if (model_) {
            llama_model_free(model_);
            model_ = nullptr;
        }
    }
}

void NpuEngine::requestCancel() {
    cancel_requested_.store(true, std::memory_order_relaxed);
}

int32_t NpuEngine::countTokens(const std::string & text) const {
    if (text.empty()) {
        return 0;
    }
    std::shared_lock<std::shared_mutex> vlock(vocab_mutex_);
    if (!vocab_) {
        return 0;
    }
    const auto toks = tokenize_string(vocab_, text, /*add_special=*/false, /*parse_special=*/true);
    return static_cast<int32_t>(toks.size());
}

GenerationStats NpuEngine::generateReply(
    const std::vector<ChatTurnMessage> & messages,
    const EngineConfig & config,
    const std::function<bool(const std::string & piece)> & on_token_delta) {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);

    GenerationStats stats{};

    if (!npu_ready_.load() || htp_devices_.empty()) {
        stats.error_message = "Генерация заблокирована: ускоритель Qualcomm Hexagon NPU не активен.";
        return stats;
    }
    if (!model_ || !ctx_ || !vocab_) {
        stats.error_message = "Сначала выберите и загрузите файл модели .gguf.";
        return stats;
    }
    if (messages.empty()) {
        stats.error_message = "Пустой запрос.";
        return stats;
    }

    cancel_requested_.store(false);

    EngineConfig cfg = config;
    cfg.sanitize();

    // Apply runtime config updates if needed
    if (cfg.n_ctx != active_config_.n_ctx ||
        cfg.n_batch != active_config_.n_batch ||
        cfg.flash_attn_mode != active_config_.flash_attn_mode ||
        cfg.kv_q8 != active_config_.kv_q8) {
        std::string ctx_err;
        if (!recreateContextLocked(cfg, &ctx_err)) {
            stats.error_message = ctx_err;
            return stats;
        }
    } else {
        if (cfg.n_threads != active_config_.n_threads) {
            llama_set_n_threads(ctx_, cfg.n_threads, cfg.n_threads);
        }
        rebuildSamplerLocked(cfg);
    }
    active_config_ = cfg;

    const int32_t n_ctx_limit = static_cast<int32_t>(llama_n_ctx(ctx_));
    stats.context_max_tokens = n_ctx_limit;

    // Reserve space for generated tokens (at least 64 tokens, up to half of context)
    const int32_t min_reserve = std::min(std::max(64, std::min(cfg.max_new_tokens, 256)), n_ctx_limit / 2);
    const int32_t max_prompt_tokens = std::max(32, n_ctx_limit - min_reserve);

    // Build message list & trim oldest turns if conversation approaches max context
    std::vector<ChatTurnMessage> working_msgs = messages;
    std::vector<llama_token> prompt_tokens;

    while (true) {
        const std::string formatted = format_chat_messages(model_, working_msgs);
        prompt_tokens = tokenize_string(vocab_, formatted, /*add_special=*/true, /*parse_special=*/true);

        if (static_cast<int32_t>(prompt_tokens.size()) <= max_prompt_tokens) {
            break;
        }

        // Find earliest removable turn (preserve optional leading system prompt + latest user message)
        const size_t first_removable = (!working_msgs.empty() && working_msgs[0].role == "system") ? 1 : 0;
        if (working_msgs.size() <= first_removable + 1) {
            // Even a single user message exceeds max_prompt_tokens — truncate leading tokens of the prompt safely
            if (static_cast<int32_t>(prompt_tokens.size()) > max_prompt_tokens) {
                prompt_tokens.erase(
                    prompt_tokens.begin(),
                    prompt_tokens.end() - max_prompt_tokens);
            }
            break;
        }
        // Drop oldest message (or user+assistant pair)
        size_t drop_count = 1;
        if (working_msgs.size() > first_removable + 2 &&
            working_msgs[first_removable].role == "user" &&
            working_msgs[first_removable + 1].role == "assistant") {
            drop_count = 2;
        }
        working_msgs.erase(
            working_msgs.begin() + static_cast<std::ptrdiff_t>(first_removable),
            working_msgs.begin() + static_cast<std::ptrdiff_t>(first_removable + drop_count));
    }

    if (prompt_tokens.empty()) {
        stats.error_message = "Не удалось токенизировать входное сообщение.";
        return stats;
    }

    stats.prompt_tokens = static_cast<int32_t>(prompt_tokens.size());

    // Prefix KV-cache reuse on Hexagon NPU
    size_t common_prefix = 0;
    const size_t max_common = std::min(session_tokens_.size(), prompt_tokens.size());
    while (common_prefix < max_common && session_tokens_[common_prefix] == prompt_tokens[common_prefix]) {
        ++common_prefix;
    }
    // Always re-evaluate at least the final token so fresh logits are available for sampling!
    if (common_prefix == prompt_tokens.size() && common_prefix > 0) {
        --common_prefix;
    }

    llama_memory_t mem = llama_get_memory(ctx_);
    if (common_prefix < session_tokens_.size()) {
        bool trimmed = false;
        if (mem) {
            if (common_prefix == 0) {
                llama_memory_clear(mem, false);
                trimmed = true;
            } else {
                trimmed = llama_memory_seq_rm(
                    mem, 0, static_cast<llama_pos>(common_prefix), -1);
            }
        }
        if (!trimmed) {
            // Hybrid / recurrent models may not support partial seq_rm — full reset
            if (mem) llama_memory_clear(mem, false);
            common_prefix = 0;
        }
        session_tokens_.resize(common_prefix);
    }

    stats.cached_tokens = static_cast<int32_t>(common_prefix);

    // Feed historical tokens to penalty sampler state
    llama_sampler_reset(sampler_);
    for (llama_token tok : prompt_tokens) {
        llama_sampler_accept(sampler_, tok);
    }

    const int32_t batch_cap = std::max<int32_t>(1, static_cast<int32_t>(llama_n_batch(ctx_)));
    ScopedBatch sbatch(batch_cap);

    // Prefill uncached prompt tokens on Hexagon NPU
    const auto t_prefill_start = std::chrono::steady_clock::now();
    const int32_t new_prompt_count = static_cast<int32_t>(prompt_tokens.size() - common_prefix);

    for (size_t offset = common_prefix; offset < prompt_tokens.size(); offset += static_cast<size_t>(batch_cap)) {
        if (cancel_requested_.load(std::memory_order_relaxed)) {
            stats.stopped_by_user     = true;
            stats.ok                  = true;
            stats.context_used_tokens = static_cast<int32_t>(session_tokens_.size());
            context_used_tokens_.store(stats.context_used_tokens);
            return stats;
        }

        const size_t chunk_end = std::min(prompt_tokens.size(), offset + static_cast<size_t>(batch_cap));
        const int32_t chunk_n  = static_cast<int32_t>(chunk_end - offset);

        sbatch.batch.n_tokens = chunk_n;
        for (int32_t i = 0; i < chunk_n; ++i) {
            const size_t global_idx = offset + static_cast<size_t>(i);
            sbatch.batch.token[i]     = prompt_tokens[global_idx];
            sbatch.batch.pos[i]       = static_cast<llama_pos>(global_idx);
            sbatch.batch.n_seq_id[i]  = 1;
            sbatch.batch.seq_id[i][0] = 0;
            sbatch.batch.logits[i]    = (global_idx + 1 == prompt_tokens.size()) ? 1 : 0;
        }

        const int32_t rc = llama_decode(ctx_, sbatch.batch);
        if (rc == 2 || cancel_requested_.load(std::memory_order_relaxed)) {
            stats.stopped_by_user     = true;
            stats.ok                  = true;
            stats.context_used_tokens = static_cast<int32_t>(session_tokens_.size());
            context_used_tokens_.store(stats.context_used_tokens);
            return stats;
        }
        if (rc != 0) {
            stats.error_message = "Ошибка декодирования промпта на Hexagon NPU (код " + std::to_string(rc) + ").";
            return stats;
        }

        for (size_t k = offset; k < chunk_end; ++k) {
            session_tokens_.push_back(prompt_tokens[k]);
        }
        context_used_tokens_.store(static_cast<int32_t>(session_tokens_.size()));
    }

    const auto t_prefill_end = std::chrono::steady_clock::now();
    const double prefill_sec = std::chrono::duration<double>(t_prefill_end - t_prefill_start).count();
    if (prefill_sec > 1e-6 && new_prompt_count > 0) {
        stats.prefill_tok_per_sec = static_cast<double>(new_prompt_count) / prefill_sec;
    }

    // Autoregressive token generation loop on Hexagon NPU
    Utf8StreamAssembler utf8_assembler;
    char piece_buf[256];
    int32_t gen_count = 0;
    const auto t_decode_start = std::chrono::steady_clock::now();

    for (int32_t step = 0; step < cfg.max_new_tokens; ++step) {
        if (cancel_requested_.load(std::memory_order_relaxed)) {
            stats.stopped_by_user = true;
            break;
        }
        if (static_cast<int32_t>(session_tokens_.size()) >= n_ctx_limit - 1) {
            // Reached max context window
            break;
        }

        const llama_token next_tok = llama_sampler_sample(sampler_, ctx_, -1);
        llama_sampler_accept(sampler_, next_tok);

        if (llama_vocab_is_eog(vocab_, next_tok)) {
            break;
        }

        int32_t piece_len = llama_token_to_piece(
            vocab_, next_tok, piece_buf, sizeof(piece_buf), 0, /*special=*/false);
        std::string piece_bytes;
        if (piece_len > static_cast<int32_t>(sizeof(piece_buf))) {
            std::vector<char> dyn_buf(static_cast<size_t>(piece_len));
            piece_len = llama_token_to_piece(
                vocab_, next_tok, dyn_buf.data(), piece_len, 0, /*special=*/false);
            if (piece_len > 0) {
                piece_bytes.assign(dyn_buf.data(), static_cast<size_t>(piece_len));
            }
        } else if (piece_len > 0) {
            piece_bytes.assign(piece_buf, static_cast<size_t>(piece_len));
        }

        const std::string ready_utf8 = utf8_assembler.push(piece_bytes.data(), piece_bytes.size());
        if (!ready_utf8.empty() && on_token_delta) {
            if (!on_token_delta(ready_utf8)) {
                stats.stopped_by_user = true;
                break;
            }
        }

        ++gen_count;

        // Advance KV cache with sampled token unless this is the final step
        if (step + 1 < cfg.max_new_tokens &&
            static_cast<int32_t>(session_tokens_.size()) < n_ctx_limit - 1 &&
            !cancel_requested_.load(std::memory_order_relaxed)) {
            sbatch.batch.n_tokens     = 1;
            sbatch.batch.token[0]     = next_tok;
            sbatch.batch.pos[0]       = static_cast<llama_pos>(session_tokens_.size());
            sbatch.batch.n_seq_id[0]  = 1;
            sbatch.batch.seq_id[0][0] = 0;
            sbatch.batch.logits[0]    = 1;

            const int32_t rc = llama_decode(ctx_, sbatch.batch);
            if (rc == 2 || cancel_requested_.load(std::memory_order_relaxed)) {
                stats.stopped_by_user = true;
                break;
            }
            if (rc != 0) {
                stats.error_message = "Ошибка генерации токена на Hexagon NPU (код " + std::to_string(rc) + ").";
                return stats;
            }
            session_tokens_.push_back(next_tok);
            context_used_tokens_.store(static_cast<int32_t>(session_tokens_.size()));
        }
    }

    const std::string tail_utf8 = utf8_assembler.flush();
    if (!tail_utf8.empty() && on_token_delta && !stats.stopped_by_user) {
        on_token_delta(tail_utf8);
    }

    const auto t_decode_end = std::chrono::steady_clock::now();
    const double decode_sec = std::chrono::duration<double>(t_decode_end - t_decode_start).count();
    if (decode_sec > 1e-6 && gen_count > 0) {
        stats.decode_tok_per_sec = static_cast<double>(gen_count) / decode_sec;
    }

    stats.ok                  = true;
    stats.generated_tokens    = gen_count;
    stats.context_used_tokens = static_cast<int32_t>(session_tokens_.size());
    context_used_tokens_.store(stats.context_used_tokens);

    return stats;
}

MemorySnapshot NpuEngine::readMemorySnapshot() const {
    MemorySnapshot snap{};

    // Read Linux/Android process physical RAM (VmRSS, VmHWM, VmSwap) from /proc/self/status
    {
        std::ifstream status_in("/proc/self/status");
        std::string line;
        while (std::getline(status_in, line)) {
            unsigned long long kb = 0;
            if (line.rfind("VmRSS:", 0) == 0) {
                if (std::sscanf(line.c_str() + 6, "%llu", &kb) == 1) {
                    snap.process_rss_mib = static_cast<double>(kb) / 1024.0;
                }
            } else if (line.rfind("VmHWM:", 0) == 0) {
                if (std::sscanf(line.c_str() + 6, "%llu", &kb) == 1) {
                    snap.process_peak_rss_mib = static_cast<double>(kb) / 1024.0;
                }
            } else if (line.rfind("VmSwap:", 0) == 0) {
                if (std::sscanf(line.c_str() + 7, "%llu", &kb) == 1) {
                    snap.process_swap_mib = static_cast<double>(kb) / 1024.0;
                }
            }
        }
    }

    // Read system total/available RAM from /proc/meminfo
    {
        std::ifstream meminfo_in("/proc/meminfo");
        std::string line;
        while (std::getline(meminfo_in, line)) {
            unsigned long long kb = 0;
            if (line.rfind("MemTotal:", 0) == 0) {
                if (std::sscanf(line.c_str() + 9, "%llu", &kb) == 1) {
                    snap.system_total_mib = static_cast<double>(kb) / 1024.0;
                }
            } else if (line.rfind("MemAvailable:", 0) == 0) {
                if (std::sscanf(line.c_str() + 13, "%llu", &kb) == 1) {
                    snap.system_avail_mib = static_cast<double>(kb) / 1024.0;
                }
            }
        }
    }

    snap.npu_htp_buffer_mib  = npu_htp_buffer_mib_.load(std::memory_order_relaxed);
    snap.host_weight_mib     = host_weight_mib_.load(std::memory_order_relaxed);
    snap.context_used_tokens = context_used_tokens_.load(std::memory_order_relaxed);
    snap.context_max_tokens  = context_max_tokens_.load(std::memory_order_relaxed);

    return snap;
}

} // namespace npu

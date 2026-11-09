#include "npu_engine.hpp"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
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

struct ScopedGguf {
    gguf_context * ctx = nullptr;
    explicit ScopedGguf(gguf_context * c) : ctx(c) {}
    ~ScopedGguf() {
        if (ctx) {
            gguf_free(ctx);
        }
    }
    ScopedGguf(const ScopedGguf &) = delete;
    ScopedGguf & operator=(const ScopedGguf &) = delete;
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

int64_t round_up_i64(int64_t v, int64_t align) {
    if (align <= 1 || v <= 0) return v;
    return ((v + align - 1) / align) * align;
}

// Type-safe GGUF integer reader — NEVER triggers GGML_ABORT on mismatched KV integer/array types
int64_t gguf_read_int64_safe(const gguf_context * ctx, const char * key, int64_t fallback) {
    if (!ctx || !key) return fallback;
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return fallback;

    const enum gguf_type t = gguf_get_kv_type(ctx, id);
    switch (t) {
        case GGUF_TYPE_UINT8:  return static_cast<int64_t>(gguf_get_val_u8(ctx, id));
        case GGUF_TYPE_INT8:   return static_cast<int64_t>(gguf_get_val_i8(ctx, id));
        case GGUF_TYPE_UINT16: return static_cast<int64_t>(gguf_get_val_u16(ctx, id));
        case GGUF_TYPE_INT16:  return static_cast<int64_t>(gguf_get_val_i16(ctx, id));
        case GGUF_TYPE_UINT32: return static_cast<int64_t>(gguf_get_val_u32(ctx, id));
        case GGUF_TYPE_INT32:  return static_cast<int64_t>(gguf_get_val_i32(ctx, id));
        case GGUF_TYPE_UINT64: return static_cast<int64_t>(gguf_get_val_u64(ctx, id));
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(ctx, id);
        case GGUF_TYPE_ARRAY: {
            // Hybrid/GQA models (e.g. Qwen3.5, Gemma3, Nemotron) may store per-layer head_count_kv arrays
            const size_t n = gguf_get_arr_n(ctx, id);
            const void * data = (n > 0) ? gguf_get_arr_data(ctx, id) : nullptr;
            if (!data || n == 0) return fallback;
            const enum gguf_type at = gguf_get_arr_type(ctx, id);
            int64_t max_val = 0;
            for (size_t i = 0; i < n; ++i) {
                int64_t v = 0;
                if (at == GGUF_TYPE_UINT32)      v = static_cast<const uint32_t *>(data)[i];
                else if (at == GGUF_TYPE_INT32)  v = static_cast<const int32_t *>(data)[i];
                else if (at == GGUF_TYPE_UINT64) v = static_cast<int64_t>(static_cast<const uint64_t *>(data)[i]);
                else if (at == GGUF_TYPE_INT64)  v = static_cast<const int64_t *>(data)[i];
                else return fallback;
                if (v > max_val) max_val = v;
            }
            return (max_val > 0) ? max_val : fallback;
        }
        default:
            return fallback;
    }
}

std::string gguf_read_str_safe(const gguf_context * ctx, const char * key) {
    if (!ctx || !key) return {};
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        return {};
    }
    const char * s = gguf_get_val_str(ctx, id);
    return s ? std::string(s) : std::string{};
}

bool htp_supports_matmul_weight_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
    }
}

bool is_k_quant_type(enum ggml_type type) {
    return type == GGML_TYPE_Q2_K ||
           type == GGML_TYPE_Q3_K ||
           type == GGML_TYPE_Q4_K ||
           type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_Q6_K;
}

// Exact HTP 2D-tiled row byte size matching ggml-hexagon/htp-repack.h & htp-tensor.h
size_t compute_htp_tiled_row_bytes(enum ggml_type type, int64_t ne0_padded) {
    switch (type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
            return static_cast<size_t>(ne0_padded) * 2u; // Repacked to FP16 on HTP
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q1_0:
            return static_cast<size_t>(ne0_padded / 32) * (576u / 32u); // 18 B / 32
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_K:
            return static_cast<size_t>(ne0_padded / 32) * (640u / 32u); // 20 B / 32
        case GGML_TYPE_Q8_0:
            return static_cast<size_t>(ne0_padded / 32) * (1088u / 32u); // 34 B / 32
        case GGML_TYPE_MXFP4:
            return static_cast<size_t>(ne0_padded / 32) * (544u / 32u); // 17 B / 32
        case GGML_TYPE_Q5_K:
            return static_cast<size_t>(ne0_padded / 32) * (768u / 32u); // 24 B / 32
        case GGML_TYPE_Q6_K:
            return static_cast<size_t>(ne0_padded / 32) * (896u / 32u); // 28 B / 32
        case GGML_TYPE_Q3_K:
            return static_cast<size_t>(ne0_padded / 32) * (512u / 32u); // 16 B / 32
        case GGML_TYPE_Q2_0:
            return static_cast<size_t>(ne0_padded / 64) * 18u;
        default:
            return ggml_row_size(type, ne0_padded);
    }
}

uint64_t compute_htp_tensor_alloc_bytes(
    enum ggml_type type,
    const int64_t * ne,
    size_t raw_bytes,
    bool is_host_input_tensor) {
    if (is_host_input_tensor || !ne) {
        return static_cast<uint64_t>(raw_bytes);
    }
    const bool will_tile =
        htp_supports_matmul_weight_type(type) &&
        ne[0] >= 32 && (ne[0] % 32 == 0) &&
        ne[3] == 1 &&
        (ggml_is_quantized(type) || ne[1] > 1);

    if (!will_tile) {
        return static_cast<uint64_t>(raw_bytes);
    }

    const int64_t ne0 = round_up_i64(ne[0], 32);
    const int64_t ne1 = round_up_i64(ne[1], 32);
    const int64_t ne2 = std::max<int64_t>(1, ne[2]);
    const int64_t ne3 = std::max<int64_t>(1, ne[3]);

    const uint64_t row_bytes  = static_cast<uint64_t>(compute_htp_tiled_row_bytes(type, ne0));
    const uint64_t tiled_size = row_bytes * static_cast<uint64_t>(ne1) *
                                static_cast<uint64_t>(ne2) * static_cast<uint64_t>(ne3);
    return std::max<uint64_t>(tiled_size, static_cast<uint64_t>(raw_bytes));
}

// Strips completed <think>...</think> blocks from historical assistant turns (per Qwen3 Jinja spec)
std::string strip_completed_think_blocks(const std::string & content) {
    std::string out = content;
    while (true) {
        const size_t start = out.find("<think>");
        if (start == std::string::npos) break;
        const size_t end = out.find("</think>", start + 7);
        if (end == std::string::npos) break;
        size_t erase_end = end + 8;
        while (erase_end < out.size() && (out[erase_end] == '\n' || out[erase_end] == '\r')) {
            ++erase_end;
        }
        out.erase(start, erase_end - start);
    }
    return out;
}

bool ends_with_directive(const std::string & text, const std::string & directive, std::string * cleaned_out) {
    size_t end = text.size();
    while (end > 0 && (text[end - 1] == ' ' || text[end - 1] == '\t' || text[end - 1] == '\n' || text[end - 1] == '\r')) {
        --end;
    }
    if (end >= directive.size() &&
        text.compare(end - directive.size(), directive.size(), directive) == 0) {
        const size_t pos = end - directive.size();
        if (pos == 0 || text[pos - 1] == ' ' || text[pos - 1] == '\n' || text[pos - 1] == '\t') {
            if (cleaned_out) {
                size_t trim_end = pos;
                while (trim_end > 0 && (text[trim_end - 1] == ' ' || text[trim_end - 1] == '\t')) {
                    --trim_end;
                }
                *cleaned_out = text.substr(0, trim_end);
            }
            return true;
        }
    }
    return false;
}

std::vector<llama_token> tokenize_prompt_smart(
    const llama_vocab * vocab,
    const std::string & formatted) {
    if (!vocab || formatted.empty()) {
        return {};
    }

    // Avoid adding a duplicate BOS token if the chat template already emitted the literal BOS text
    // (e.g. Llama 3 "<|begin_of_text|>" or Gemma "<bos>")
    bool add_special = llama_vocab_get_add_bos(vocab);
    if (add_special) {
        const llama_token bos_id = llama_vocab_bos(vocab);
        if (bos_id != LLAMA_TOKEN_NULL) {
            const char * bos_txt = llama_vocab_get_text(vocab, bos_id);
            if (bos_txt && bos_txt[0] != '\0') {
                const size_t blen = std::strlen(bos_txt);
                if (formatted.size() >= blen && formatted.compare(0, blen, bos_txt) == 0) {
                    add_special = false;
                }
            }
        }
    }

    int32_t n_est = static_cast<int32_t>(formatted.size()) + 16;
    std::vector<llama_token> tokens(static_cast<size_t>(n_est));
    int32_t n = llama_tokenize(
        vocab,
        formatted.data(),
        static_cast<int32_t>(formatted.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        add_special,
        /*parse_special=*/true);
    if (n < 0 && n != INT32_MIN) {
        tokens.resize(static_cast<size_t>(-n));
        n = llama_tokenize(
            vocab,
            formatted.data(),
            static_cast<int32_t>(formatted.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            add_special,
            /*parse_special=*/true);
    }
    if (n < 0) {
        return {};
    }
    tokens.resize(static_cast<size_t>(n));
    return tokens;
}

std::vector<llama_token> tokenize_plain_text(
    const llama_vocab * vocab,
    const std::string & text) {
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
        /*add_special=*/false,
        /*parse_special=*/true);
    if (n < 0 && n != INT32_MIN) {
        tokens.resize(static_cast<size_t>(-n));
        n = llama_tokenize(
            vocab,
            text.data(),
            static_cast<int32_t>(text.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            /*add_special=*/false,
            /*parse_special=*/true);
    }
    if (n < 0) {
        return {};
    }
    tokens.resize(static_cast<size_t>(n));
    return tokens;
}

bool check_logits_finite(llama_context * ctx, const llama_vocab * vocab) {
    if (!ctx || !vocab) return false;
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) return false;
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) return false;

    const int32_t head_check = std::min<int32_t>(n_vocab, 64);
    for (int32_t i = 0; i < head_check; ++i) {
        if (!std::isfinite(logits[i])) {
            return false;
        }
    }
    const int32_t step = std::max<int32_t>(1, n_vocab / 256);
    for (int32_t i = head_check; i < n_vocab; i += step) {
        if (!std::isfinite(logits[i])) {
            return false;
        }
    }
    return true;
}

std::string format_mib_or_gib(double mib) {
    std::ostringstream oss;
    if (mib >= 1024.0) {
        oss << std::fixed << std::setprecision(2) << (mib / 1024.0) << " ГБ ("
            << std::setprecision(0) << mib << " МБ)";
    } else {
        oss << std::fixed << std::setprecision(0) << mib << " МБ";
    }
    return oss.str();
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
            pending_.replace(i, 1, "\xEF\xBF\xBD");
            return push("", 0);
        }
        if (i + seq_len > total) {
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
// Chat Template Formatter (including Qwen3 / Qwen3.5 <think> support)
// ---------------------------------------------------------------------------

std::string NpuEngine::formatChatPromptWithTemplate(
    const char * raw_template,
    const std::vector<ChatTurnMessage> & messages,
    bool enable_thinking) {
    const std::string tmpl_str = raw_template ? raw_template : "";
    const bool is_qwen3_chatml_think =
        tmpl_str.find("<|im_start|>") != std::string::npos &&
        tmpl_str.find("<think>") != std::string::npos;

    // Sanitize messages: strip completed <think>...</think> from prior assistant turns
    // and check for explicit /think or /no_think directives on the latest user turn.
    std::vector<ChatTurnMessage> clean_msgs = messages;
    bool think_for_turn = enable_thinking;

    for (size_t i = 0; i < clean_msgs.size(); ++i) {
        if (clean_msgs[i].role == "assistant") {
            clean_msgs[i].content = strip_completed_think_blocks(clean_msgs[i].content);
        }
    }
    if (!clean_msgs.empty() && clean_msgs.back().role == "user") {
        std::string stripped;
        if (ends_with_directive(clean_msgs.back().content, "/think", &stripped)) {
            think_for_turn = true;
            if (!stripped.empty()) clean_msgs.back().content = stripped;
        } else if (ends_with_directive(clean_msgs.back().content, "/no_think", &stripped)) {
            think_for_turn = false;
            if (!stripped.empty()) clean_msgs.back().content = stripped;
        }
    }

    if (is_qwen3_chatml_think) {
        std::ostringstream oss;
        for (const auto & m : clean_msgs) {
            oss << "<|im_start|>" << m.role << "\n" << m.content << "<|im_end|>\n";
        }
        if (think_for_turn) {
            oss << "<|im_start|>assistant\n<think>\n";
        } else {
            // Official Qwen3 non-thinking prefill sequence: prevents broken/unbounded CoT on 0.6B–4B models!
            oss << "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        }
        return oss.str();
    }

    std::vector<llama_chat_message> c_msgs;
    c_msgs.reserve(clean_msgs.size());
    size_t approx_chars = 64;
    for (const auto & m : clean_msgs) {
        c_msgs.push_back(llama_chat_message{m.role.c_str(), m.content.c_str()});
        approx_chars += m.role.size() + m.content.size() + 32;
    }

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

    int32_t written = try_apply(raw_template);
    if (written < 0) {
        written = try_apply("chatml");
    }

    if (written >= 0 && static_cast<size_t>(written) <= buf.size()) {
        return std::string(buf.data(), static_cast<size_t>(written));
    }

    std::ostringstream oss;
    for (const auto & m : clean_msgs) {
        oss << "<|im_start|>" << m.role << "\n" << m.content << "<|im_end|>\n";
    }
    oss << "<|im_start|>assistant\n";
    return oss.str();
}

// ---------------------------------------------------------------------------
// Zero-Allocation GGUF & Hexagon HTP Capacity Gate
// ---------------------------------------------------------------------------

GgufPreflightReport NpuEngine::evaluateGgufForHexagonNpu(
    const gguf_context * gctx,
    uint64_t file_size_bytes,
    const EngineConfig & cfg,
    int32_t hexagon_arch,
    int32_t active_htp_devices,
    double effective_avail_ram_mib,
    double system_total_ram_mib) {
    GgufPreflightReport rep{};

    if (!gctx) {
        rep.error_message =
            "Не удалось разобрать структуру метаданных GGUF. Файл повреждён, обрезан или не является корректной моделью .gguf.";
        return rep;
    }

    const int64_t n_tensors = gguf_get_n_tensors(gctx);
    if (n_tensors <= 0) {
        rep.error_message =
            "В выбранном файле .gguf отсутствуют тензоры весов (возможно, это файл только словаря или адаптер LoRA).";
        return rep;
    }

    rep.arch_name = gguf_read_str_safe(gctx, "general.architecture");
    const std::string model_name = gguf_read_str_safe(gctx, "general.name");
    const std::string size_label = gguf_read_str_safe(gctx, "general.size_label");

    int64_t n_layer   = 0;
    int64_t n_embd    = 0;
    int64_t n_head    = 0;
    int64_t n_head_kv = 0;
    int64_t d_k       = 0;
    int64_t d_v       = 0;

    if (!rep.arch_name.empty()) {
        n_layer   = gguf_read_int64_safe(gctx, (rep.arch_name + ".block_count").c_str(), 0);
        n_embd    = gguf_read_int64_safe(gctx, (rep.arch_name + ".embedding_length").c_str(), 0);
        n_head    = gguf_read_int64_safe(gctx, (rep.arch_name + ".attention.head_count").c_str(), 0);
        n_head_kv = gguf_read_int64_safe(gctx, (rep.arch_name + ".attention.head_count_kv").c_str(), n_head);
        const int64_t default_hd = (n_head > 0 && n_embd > 0) ? (n_embd / n_head) : 128;
        d_k       = gguf_read_int64_safe(gctx, (rep.arch_name + ".attention.key_length").c_str(), default_hd);
        d_v       = gguf_read_int64_safe(gctx, (rep.arch_name + ".attention.value_length").c_str(), d_k);
    }
    rep.n_layers = static_cast<int32_t>(n_layer);

    uint64_t raw_total_bytes      = 0;
    uint64_t htp_tiled_bytes      = 0;
    uint64_t host_embd_bytes      = 0;
    uint64_t matrix_2d_raw_bytes  = 0;
    uint64_t k_quant_raw_bytes    = 0;
    uint64_t unsupp_quant_bytes   = 0;
    std::map<enum ggml_type, uint64_t> bytes_by_type;
    std::string first_unsupp_type;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * tname_c = gguf_get_tensor_name(gctx, i);
        const std::string tname = tname_c ? tname_c : "";
        const enum ggml_type ttype = gguf_get_tensor_type(gctx, i);
        const size_t tsize = gguf_get_tensor_size(gctx, i);
        const int64_t * ne = gguf_get_tensor_ne(gctx, i);

        raw_total_bytes += static_cast<uint64_t>(tsize);
        bytes_by_type[ttype] += static_cast<uint64_t>(tsize);

        const bool is_tok_embd =
            (tname == "token_embd.weight" || tname == "tok_embeddings.weight" ||
             tname.find("per_layer_token_embd") != std::string::npos);

        if (is_tok_embd) {
            host_embd_bytes += static_cast<uint64_t>(tsize);
        } else {
            htp_tiled_bytes += compute_htp_tensor_alloc_bytes(ttype, ne, tsize, false);
        }

        const bool is_2d_weight = ne && ne[0] >= 32 && ne[1] > 1 && !is_tok_embd;
        if (is_2d_weight) {
            matrix_2d_raw_bytes += static_cast<uint64_t>(tsize);
            if (is_k_quant_type(ttype)) {
                k_quant_raw_bytes += static_cast<uint64_t>(tsize);
            }
            if (!htp_supports_matmul_weight_type(ttype)) {
                unsupp_quant_bytes += static_cast<uint64_t>(tsize);
                if (first_unsupp_type.empty()) {
                    const char * tn = ggml_type_name(ttype);
                    first_unsupp_type = tn ? tn : "IQ/TQ";
                }
            }
        }
    }

    if (file_size_bytes > raw_total_bytes && raw_total_bytes == 0) {
        raw_total_bytes = file_size_bytes;
    }

    // Dominant weight quantization label
    uint64_t best_type_bytes = 0;
    enum ggml_type dom_type  = GGML_TYPE_F16;
    for (const auto & kv : bytes_by_type) {
        if (kv.second > best_type_bytes) {
            best_type_bytes = kv.second;
            dom_type        = kv.first;
        }
    }
    const char * dom_name = ggml_type_name(dom_type);
    rep.dominant_quant    = dom_name ? dom_name : "GGUF";

    constexpr double kBytesPerMib = 1024.0 * 1024.0;
    rep.raw_weight_bytes      = raw_total_bytes;
    rep.htp_tiled_weights_mib = static_cast<double>(htp_tiled_bytes) / kBytesPerMib;
    rep.host_embd_mib         = static_cast<double>(host_embd_bytes) / kBytesPerMib;
    rep.has_k_quants          = (k_quant_raw_bytes > 0);
    rep.has_unsupported_quants =
        (matrix_2d_raw_bytes > 0 && unsupp_quant_bytes * 4 >= matrix_2d_raw_bytes);

    // Estimate KV-cache footprint on HTP0
    const int64_t eff_layers  = (n_layer > 0) ? n_layer : 28;
    const int64_t eff_kv_head = (n_head_kv > 0) ? n_head_kv : 8;
    const int64_t eff_dk      = (d_k > 0) ? d_k : 128;
    const int64_t eff_dv      = (d_v > 0) ? d_v : 128;
    const double  bytes_per_kv_elem = cfg.kv_q8 ? (34.0 / 32.0) : 2.0;
    const double  kv_bytes =
        static_cast<double>(eff_layers) *
        static_cast<double>(std::max<int32_t>(256, cfg.n_ctx)) *
        static_cast<double>(eff_kv_head) *
        static_cast<double>(eff_dk + eff_dv) *
        bytes_per_kv_elem;

    rep.kv_cache_mib        = kv_bytes / kBytesPerMib;
    rep.compute_reserve_mib = 300.0; // HTP dynamic compute buffer + skel VA + rpcmem chunk alignment
    const double total_weights_mib = rep.htp_tiled_weights_mib + rep.host_embd_mib;
    rep.total_required_npu_mib = total_weights_mib + rep.kv_cache_mib + rep.compute_reserve_mib;
    rep.effective_avail_ram_mib = effective_avail_ram_mib;

    const int32_t arch = (hexagon_arch > 0) ? hexagon_arch : 73;
    const int32_t ndev = (arch < 75) ? 1 : std::max<int32_t>(1, active_htp_devices);

    if (arch <= 79) {
        // 32-bit Hexagon cDSP virtual address space (v73 = 3000 MiB, v75/v79 = 3200 MiB per HTP session)
        const double per_dev_vmem_mib = (arch >= 75) ? 3200.0 : 3000.0;
        rep.npu_vmem_ceiling_mib      = per_dev_vmem_mib * static_cast<double>(ndev);
        rep.npu_weight_ceiling_mib    = (per_dev_vmem_mib - 520.0) * static_cast<double>(ndev);
    } else {
        // 64-bit DMA64 on Hexagon v81+ — bounded by physical RAM
        const double sys_cap = (system_total_ram_mib > 0.0) ? (system_total_ram_mib * 0.70) : 8192.0;
        rep.npu_vmem_ceiling_mib   = sys_cap;
        rep.npu_weight_ceiling_mib = std::max(1024.0, sys_cap - 520.0);
    }

    // GATE 1: Reject quantizations that Hexagon HTP MUL_MAT cannot execute natively (would force forbidden CPU fallback)
    if (rep.has_unsupported_quants) {
        std::ostringstream err;
        err << "Формат квантования «" << first_unsupp_type << "» не поддерживается аппаратным ядром Qualcomm Hexagon HTP NPU, "
            << "а фоллбэк на CPU/GPU запрещён.\n\n"
            << "Рекомендуемые форматы для Hexagon NPU:\n"
            << "• Q4_0 (int4, симметричный 4-бит — оптимальный по скорости и памяти)\n"
            << "• Q8_0 (int8 — максимальная точность для компактных моделей 0.5B–1.7B)";
        rep.error_message = err.str();
        return rep;
    }

    // GATE 2: Reject oversized models whose weights exceed Hexagon NPU cDSP VA space (e.g. 7B / 8B / 9B / 14B)
    if (total_weights_mib > rep.npu_weight_ceiling_mib) {
        std::ostringstream err;
        err << "Модель слишком велика для размещения в Qualcomm Hexagon NPU";
        if (!size_label.empty()) {
            err << " (" << size_label << ")";
        } else if (!model_name.empty()) {
            err << " («" << model_name << "»)";
        }
        err << ".\n\n"
            << "• Объём весов модели для NPU: " << format_mib_or_gib(total_weights_mib)
            << " [" << rep.dominant_quant << "]\n"
            << "• Аппаратный потолок весов Hexagon HTP v" << arch << ": "
            << format_mib_or_gib(rep.npu_weight_ceiling_mib)
            << " (32-бит адресное пространство cDSP " << static_cast<int>(rep.npu_vmem_ceiling_mib) << " МБ)\n\n"
            << "Выгрузка модели в NPU остановлена до выделения DMA-памяти (вылет приложения предотвращён).\n"
            << "Используйте модели объёмом 0.5B–3B (до 4B в чистом Q4_0) с квантованием Q4_0 (int4) или Q8_0.";
        rep.error_message = err.str();
        return rep;
    }

    // GATE 3: Weights fit, but Weights + KV-cache + Compute exceed Hexagon cDSP VA budget
    if (rep.total_required_npu_mib > rep.npu_vmem_ceiling_mib) {
        std::ostringstream err;
        err << "Модель (" << format_mib_or_gib(total_weights_mib) << ") и KV-кэш контекста "
            << cfg.n_ctx << " токенов (" << format_mib_or_gib(rep.kv_cache_mib)
            << ") суммарно требуют " << format_mib_or_gib(rep.total_required_npu_mib)
            << ", что превышает лимит адресного пространства Hexagon HTP v" << arch << " ("
            << format_mib_or_gib(rep.npu_vmem_ceiling_mib) << ").\n\n"
            << "Откройте «Параметры», уменьшите макс. контекст (например, до 2048) и включите квантование KV-кэша Q8_0.";
        rep.error_message = err.str();
        return rep;
    }

    // GATE 4: Protect against Android lmkd SIGKILL when physical MemAvailable is insufficient for pinned rpcmem DMA-BUF
    constexpr double kAndroidLmkdReserveMib = 700.0;
    if (effective_avail_ram_mib > 0.0 &&
        (rep.total_required_npu_mib + kAndroidLmkdReserveMib) > effective_avail_ram_mib) {
        std::ostringstream err;
        err << "Недостаточно свободной оперативной памяти для закрепления буферов Hexagon NPU (RPCMEM DMA-BUF).\n\n"
            << "• Требуется для модели и KV-кэша: ~" << format_mib_or_gib(rep.total_required_npu_mib) << "\n"
            << "• Безопасный резерв ОС Android: " << static_cast<int>(kAndroidLmkdReserveMib) << " МБ\n"
            << "• Свободно ОЗУ сейчас: " << format_mib_or_gib(effective_avail_ram_mib) << "\n\n"
            << "Загрузка остановлена до выделения памяти, чтобы предотвратить принудительное завершение процесса системой Android. "
            << "Закройте фоновые приложения, уменьшите n_ctx или выберите модель меньшего размера (0.5B–1.7B Q4_0).";
        rep.error_message = err.str();
        return rep;
    }

    if (rep.has_k_quants) {
        rep.quant_warning =
            "Обнаружено K-квантование (" + rep.dominant_quant +
            "): включён режим совместимости NPU. Для максимальной скорости и экономии памяти Hexagon HTP рекомендуется Q4_0 (int4) или Q8_0.";
    }

    rep.ok = true;
    return rep;
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

        // 3. Parse offloaded layers ("offloaded 29/29 layers to GPU")
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
    const std::string & power_mode,
    bool allow_fp16_hmx) {
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
    ::setenv("GGML_HEXAGON_HOSTBUF",    "1",                   1);

    // UPSTREAM BUGFIX #31933: Never set GGML_HEXAGON_OPPOLL=1 on Android (causes zero-timeout dspqueue_read 0xc logd flood)
    ::setenv("GGML_HEXAGON_OPPOLL",     "0",                   1);

    // UPSTREAM BUGFIX #31476: HTP_OP_GET_ROWS corrupts dmin on Q4_K and Q5_K token embeddings (e.g. Qwen3-0.6B-Q4_K_M).
    // token_embd.weight already resides in zero-copy HTP0-HOST RPCMEM; filtering GET_ROWS avoids embedding corruption
    // while keeping 100% of Transformer layers executing on HTP0!
    ::setenv("GGML_HEXAGON_OPFILTER",   "GET_ROWS",            1);

    // UPSTREAM BUGFIX #31924: Force HVX Flash Attention on HTP0 to avoid HMX FA per-head GQA mask VTCM overwrite at pos >= 129
    ::setenv("GGML_HEXAGON_FA_SELECT",  "1",                   1);

    // UPSTREAM BUGFIX #31645 / #29473 / #29740:
    // Hexagon v73 sub-flagship SoCs (Snapdragon 8s Gen 3 SM8635, 7 Gen 4 SM7750, 7+ Gen 3 SM7675, 6 Gen 4 SM6650)
    // do NOT support FP16 HMX matmul/GDN (stock FP16 HMX returns inf/NaN on prefill m >= 5).
    // Setting NHMX=0 executes 100% of NPU ops on the 4x HVX vector units inside HTP0 with identical decode tok/s and 100% accuracy!
    if (!allow_fp16_hmx) {
        ::setenv("GGML_HEXAGON_NHMX",       "0", 1);
        ::setenv("GGML_HEXAGON_MM_SELECT",  "1", 1);
        ::setenv("GGML_HEXAGON_GDN_SELECT", "1", 1);
        ::setenv("GGML_HEXAGON_LI_SELECT",  "1", 1);
        hmx_enabled_ = false;
    } else {
        ::setenv("GGML_HEXAGON_NHMX",       "1", 1);
        ::setenv("GGML_HEXAGON_MM_SELECT",  "2", 1);
        ::setenv("GGML_HEXAGON_GDN_SELECT", "2", 1);
        ::setenv("GGML_HEXAGON_LI_SELECT",  "2", 1);
        hmx_enabled_ = true;
    }

    // Defense-in-depth: block all ops on OpenCL Adreno GPU even if libggml-opencl.so is in memory
    ::setenv("GGML_OPENCL_OPFILTER", ".*", 1);

    llama_log_set(llama_log_trampoline, nullptr);

    try {
        if (!backend_initialized_.load()) {
            if (!native_lib_dir.empty()) {
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
    } catch (...) {
        npu_ready_.store(false);
        res.ok = false;
        res.error_message = "Неизвестное исключение при инициализации Qualcomm Hexagon NPU.";
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
    res.hmx_enabled    = hmx_enabled_ && (hmx > 0);

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
    if (hvx > 0 || vtcm > 0) {
        hw << " • HVX: " << hvx;
        if (res.hmx_enabled) hw << " • HMX: FP16";
        else hw << " • Режим точности HVX";
        if (vtcm > 0) hw << " • VTCM: " << vtcm << " МБ";
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
    out.kept_previous_model = model_loaded_.load();

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
    if (std::fseek(fp, 0, SEEK_END) != 0) {
        out.error_message = "Выбранный источник файла не поддерживает произвольное позиционирование (seek).";
        return out;
    }
    const long end_pos = std::ftell(fp);
    const uint64_t file_bytes = (end_pos > 0) ? static_cast<uint64_t>(end_pos) : 0ULL;
    if (std::fseek(fp, 0, SEEK_SET) != 0) {
        out.error_message = "Не удалось перемотать поток чтения файла .gguf в начало.";
        return out;
    }

    EngineConfig cfg = config;
    cfg.sanitize();

    // HARD GATE 2 (ZERO-ALLOCATION PREFLIGHT): Parse GGUF metadata & check Hexagon HTP vmem + Android RAM
    // BEFORE unloading any active model or allocating any rpcmem DMA-BUF chunks!
    GgufPreflightReport preflight{};
    try {
        gguf_init_params gparams{ /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        ScopedGguf gctx(gguf_init_from_file_ptr(fp, gparams));

        const MemorySnapshot mem = readMemorySnapshot();
        const double reclaimable_mib = npu_htp_buffer_mib_.load() + host_weight_mib_.load();
        const double effective_avail =
            (mem.system_avail_mib > 0.0) ? (mem.system_avail_mib + reclaimable_mib) : 0.0;

        int32_t arch = 73;
        {
            std::lock_guard<std::mutex> lk(log_parse_mutex_);
            if (parsed_hexagon_arch_ > 0) arch = parsed_hexagon_arch_;
        }

        preflight = evaluateGgufForHexagonNpu(
            gctx.ctx,
            file_bytes,
            cfg,
            arch,
            static_cast<int32_t>(htp_devices_.size()),
            effective_avail,
            mem.system_total_mib);
    } catch (const std::exception & e) {
        out.error_message = std::string("Ошибка предварительного анализа GGUF: ") + e.what();
        return out;
    } catch (...) {
        out.error_message = "Неизвестная ошибка при проверке структуры GGUF файла.";
        return out;
    }

    if (!preflight.ok) {
        NPU_LOGW("Rejected GGUF before NPU allocation: %s", preflight.error_message.c_str());
        out.error_message = preflight.error_message;
        return out;
    }

    if (std::fseek(fp, 0, SEEK_SET) != 0) {
        out.error_message = "Ошибка позиционирования потока GGUF после проверки метаданных.";
        return out;
    }

    // Now safe to unload any previous model and allocate new Hexagon HTP buffers
    unloadModelLocked();
    out.kept_previous_model = false;
    cancel_requested_.store(false);

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
    mparams.n_gpu_layers = cfg.n_npu_layers;     // Always >= 1 (default 999 = all layers on Hexagon NPU)
    mparams.load_mode    = LLAMA_LOAD_MODE_NONE; // --no-mmap: required for 2D HMX/HVX tile repacking in RPCMEM
    mparams.split_mode   = (htp_devices_.size() > 1) ? LLAMA_SPLIT_MODE_LAYER : LLAMA_SPLIT_MODE_NONE;

    ProgressCallbackBridge prog_bridge{&on_progress, &cancel_requested_};
    mparams.progress_callback           = model_load_progress_cb;
    mparams.progress_callback_user_data = &prog_bridge;

    llama_model * loaded = nullptr;
    try {
        loaded = llama_model_load_from_file_ptr(fp, mparams);
    } catch (const std::exception & e) {
        out.error_message =
            std::string("Не удалось разместить веса модели в памяти Qualcomm Hexagon NPU: ") + e.what();
        return out;
    } catch (...) {
        out.error_message = "Критическое исключение при загрузке весов GGUF в Hexagon NPU.";
        return out;
    }

    return finishModelLoadLocked(loaded, cfg, preflight.quant_warning);
}

ModelLoadResult NpuEngine::loadModelFromPath(
    const std::string & file_path,
    const EngineConfig & config,
    const std::function<void(float)> & on_progress) {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);

    ModelLoadResult out{};
    out.kept_previous_model = model_loaded_.load();

    if (!npu_ready_.load() || htp_devices_.empty()) {
        out.error_message =
            "Загрузка модели отклонена: Qualcomm Hexagon NPU (HTP0) не активен. "
            "Запуск модели на CPU или GPU строго заблокирован.";
        return out;
    }

    std::ifstream ifs(file_path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        out.error_message = "Не удалось открыть файл модели: " + file_path;
        return out;
    }
    const auto end_pos = ifs.tellg();
    const uint64_t file_bytes = (end_pos > 0) ? static_cast<uint64_t>(end_pos) : 0ULL;
    ifs.seekg(0, std::ios::beg);

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

    EngineConfig cfg = config;
    cfg.sanitize();

    // ZERO-ALLOCATION PREFLIGHT CHECK
    GgufPreflightReport preflight{};
    try {
        gguf_init_params gparams{ /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        ScopedGguf gctx(gguf_init_from_file(file_path.c_str(), gparams));

        const MemorySnapshot mem = readMemorySnapshot();
        const double reclaimable_mib = npu_htp_buffer_mib_.load() + host_weight_mib_.load();
        const double effective_avail =
            (mem.system_avail_mib > 0.0) ? (mem.system_avail_mib + reclaimable_mib) : 0.0;

        int32_t arch = 73;
        {
            std::lock_guard<std::mutex> lk(log_parse_mutex_);
            if (parsed_hexagon_arch_ > 0) arch = parsed_hexagon_arch_;
        }

        preflight = evaluateGgufForHexagonNpu(
            gctx.ctx,
            file_bytes,
            cfg,
            arch,
            static_cast<int32_t>(htp_devices_.size()),
            effective_avail,
            mem.system_total_mib);
    } catch (const std::exception & e) {
        out.error_message = std::string("Ошибка предварительного анализа GGUF: ") + e.what();
        return out;
    } catch (...) {
        out.error_message = "Неизвестная ошибка при проверке структуры GGUF файла.";
        return out;
    }

    if (!preflight.ok) {
        NPU_LOGW("Rejected GGUF before NPU allocation: %s", preflight.error_message.c_str());
        out.error_message = preflight.error_message;
        return out;
    }

    unloadModelLocked();
    out.kept_previous_model = false;
    cancel_requested_.store(false);

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
        out.error_message =
            std::string("Не удалось разместить веса модели в памяти Qualcomm Hexagon NPU: ") + e.what();
        return out;
    } catch (...) {
        out.error_message = "Критическое исключение при загрузке весов GGUF в Hexagon NPU.";
        return out;
    }

    return finishModelLoadLocked(loaded, cfg, preflight.quant_warning);
}

ModelLoadResult NpuEngine::finishModelLoadLocked(
    llama_model * loaded_model,
    const EngineConfig & sanitized_cfg,
    const std::string & preflight_quant_warning) {
    ModelLoadResult out{};

    if (!loaded_model) {
        if (cancel_requested_.load()) {
            out.error_message = "Загрузка модели отменена пользователем.";
        } else {
            std::lock_guard<std::mutex> lk(log_parse_mutex_);
            out.error_message =
                "Не удалось выделить буферы Qualcomm Hexagon NPU для выбранной модели GGUF. "
                "Используйте модель меньшего размера (0.5B–3B) с квантованием Q4_0 (int4) или Q8_0.";
            if (!last_hex_error_.empty()) {
                out.error_message += "\nДетали драйвера: " + last_hex_error_;
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

    // HARD GATE 3: Verify weights were actually placed onto Hexagon HTP buffers!
    if (log_npu_layers <= 0 && htp_mib <= 0.01) {
        llama_model_free(loaded_model);
        out.error_message =
            "ОШИБКА NPU-ВЕРИФИКАЦИИ: Ни один слой модели не был размещён в буферах Hexagon HTP0! "
            "Запуск чисто на CPU заблокирован. Используйте модель с нативным квантованием NPU: Q4_0 (int4) или Q8_0.";
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
    out.quant_warning   = preflight_quant_warning;

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
    cparams.op_offload      = true; // Offload tensor ops to Hexagon NPU (except GET_ROWS via OPFILTER)!
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
    } catch (...) {
        if (err_out) {
            *err_out = "Неизвестное исключение при выделении контекста на Hexagon NPU.";
        }
        return false;
    }

    if (!ctx_) {
        if (err_out) {
            *err_out = "Не удалось выделить KV-кэш контекста (" + std::to_string(cfg.n_ctx) +
                       " токенов) в памяти Hexagon NPU. Уменьшите макс. контекст до 2048 или включите Q8_0 KV-кэш.";
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
        if (cfg.enable_thinking != active_config_.enable_thinking) {
            // Reset prefix cache cleanly if Qwen3 <think> mode changed
            llama_memory_t mem = llama_get_memory(ctx_);
            if (mem) llama_memory_clear(mem, false);
            session_tokens_.clear();
            context_used_tokens_.store(0);
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
    const auto toks = tokenize_plain_text(vocab_, text);
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

    const int32_t min_reserve = std::min(std::max(64, std::min(cfg.max_new_tokens, 256)), n_ctx_limit / 2);
    const int32_t max_prompt_tokens = std::max(32, n_ctx_limit - min_reserve);

    const char * raw_tmpl = llama_model_chat_template(model_, nullptr);
    std::vector<ChatTurnMessage> working_msgs = messages;
    std::vector<llama_token> prompt_tokens;

    while (true) {
        const std::string formatted =
            formatChatPromptWithTemplate(raw_tmpl, working_msgs, cfg.enable_thinking);
        prompt_tokens = tokenize_prompt_smart(vocab_, formatted);

        if (static_cast<int32_t>(prompt_tokens.size()) <= max_prompt_tokens) {
            break;
        }

        const size_t first_removable = (!working_msgs.empty() && working_msgs[0].role == "system") ? 1 : 0;
        if (working_msgs.size() <= first_removable + 1) {
            if (static_cast<int32_t>(prompt_tokens.size()) > max_prompt_tokens) {
                prompt_tokens.erase(
                    prompt_tokens.begin(),
                    prompt_tokens.end() - max_prompt_tokens);
            }
            break;
        }

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
            if (mem) llama_memory_clear(mem, false);
            common_prefix = 0;
        }
        session_tokens_.resize(common_prefix);
    }

    stats.cached_tokens = static_cast<int32_t>(common_prefix);

    llama_sampler_reset(sampler_);
    for (llama_token tok : prompt_tokens) {
        llama_sampler_accept(sampler_, tok);
    }

    const int32_t batch_cap = std::max<int32_t>(1, static_cast<int32_t>(llama_n_batch(ctx_)));
    ScopedBatch sbatch(batch_cap);

    // Prefill uncached prompt tokens on Hexagon NPU (wrapped in try/catch to guard lazy fastrpc_mmap!)
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

        int32_t rc = -1;
        try {
            rc = llama_decode(ctx_, sbatch.batch);
        } catch (const std::exception & e) {
            if (mem) llama_memory_clear(mem, false);
            session_tokens_.clear();
            context_used_tokens_.store(0);
            stats.error_message =
                std::string("Ошибка аппаратного маппинга/выполнения на Hexagon NPU (fastrpc): ") + e.what() +
                ". Попробуйте уменьшить макс. контекст или использовать более компактную модель Q4_0.";
            return stats;
        } catch (...) {
            if (mem) llama_memory_clear(mem, false);
            session_tokens_.clear();
            context_used_tokens_.store(0);
            stats.error_message = "Критическое исключение драйвера Hexagon NPU при обработке промпта.";
            return stats;
        }

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
            break;
        }

        // Guard against NaN/Inf logits before sampling
        if (!check_logits_finite(ctx_, vocab_)) {
            if (mem) llama_memory_clear(mem, false);
            session_tokens_.clear();
            context_used_tokens_.store(0);
            stats.error_message =
                "Обнаружено переполнение логитов (NaN/Inf) на NPU. Рекомендуется использовать модель в формате Q4_0 (int4) или Q8_0.";
            return stats;
        }

        llama_token next_tok = LLAMA_TOKEN_NULL;
        try {
            next_tok = llama_sampler_sample(sampler_, ctx_, -1);
            llama_sampler_accept(sampler_, next_tok);
        } catch (const std::exception & e) {
            stats.error_message = std::string("Ошибка сэмплирования токена: ") + e.what();
            return stats;
        }

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

        if (step + 1 < cfg.max_new_tokens &&
            static_cast<int32_t>(session_tokens_.size()) < n_ctx_limit - 1 &&
            !cancel_requested_.load(std::memory_order_relaxed)) {
            sbatch.batch.n_tokens     = 1;
            sbatch.batch.token[0]     = next_tok;
            sbatch.batch.pos[0]       = static_cast<llama_pos>(session_tokens_.size());
            sbatch.batch.n_seq_id[0]  = 1;
            sbatch.batch.seq_id[0][0] = 0;
            sbatch.batch.logits[0]    = 1;

            int32_t rc = -1;
            try {
                rc = llama_decode(ctx_, sbatch.batch);
            } catch (const std::exception & e) {
                stats.error_message =
                    std::string("Ошибка выполнения шага генерации на Hexagon NPU: ") + e.what();
                return stats;
            } catch (...) {
                stats.error_message = "Критическое исключение Hexagon NPU при генерации токена.";
                return stats;
            }

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

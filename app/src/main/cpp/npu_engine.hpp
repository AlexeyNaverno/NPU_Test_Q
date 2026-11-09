#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;
struct llama_sampler;
struct llama_vocab;
struct ggml_backend_device;

namespace npu {

struct EngineConfig {
    int32_t n_ctx             = 4096;
    int32_t max_new_tokens    = 512;
    int32_t n_batch           = 512;
    int32_t n_threads         = 6;
    int32_t n_npu_layers      = 999; // Always >= 1 (999 = offload all layers to Hexagon NPU)
    float   temperature       = 0.7f;
    float   top_p             = 0.9f;
    int32_t top_k             = 40;
    float   min_p             = 0.05f;
    float   repeat_penalty    = 1.10f;
    int32_t repeat_last_n     = 64;
    int32_t seed              = -1;  // < 0 -> random LLAMA_DEFAULT_SEED
    int32_t flash_attn_mode   = 1;   // -1 = auto, 0 = disabled, 1 = enabled (default ON for HTP)
    bool    kv_q8             = false; // false = F16, true = Q8_0 KV cache
    std::string power_mode    = "sustained_high_performance";
    std::string htp_devices   = "HTP0";
    std::string system_prompt = "";

    void sanitize();
};

struct InitResult {
    bool        ok              = false;
    int32_t     hexagon_arch    = 0;   // e.g. 73 on Snapdragon 8s Gen 3 (SM8635)
    int32_t     htp_device_cnt  = 0;
    bool        has_v73_skel    = false;
    std::string active_devices;        // e.g. "HTP0 (Hexagon)"
    std::string hw_summary;            // e.g. "Hexagon HTP v73 • HVX: 4 • HMX: 1 • VTCM: 8 MB"
    std::string error_message;
};

struct ModelLoadResult {
    bool        ok                = false;
    std::string model_desc;
    std::string quant_name;
    uint64_t    model_bytes       = 0;
    uint64_t    model_params      = 0;
    int32_t     n_ctx_train       = 0;
    int32_t     n_ctx_active      = 0;
    int32_t     n_layers_total    = 0;
    int32_t     n_layers_npu      = 0;
    double      htp_buffer_mib    = 0.0;
    double      host_buffer_mib   = 0.0;
    std::string npu_device_used;
    std::string error_message;
};

struct ChatTurnMessage {
    std::string role;    // "system", "user", "assistant"
    std::string content;
};

struct GenerationStats {
    bool        ok                  = false;
    bool        stopped_by_user     = false;
    int32_t     prompt_tokens       = 0;
    int32_t     cached_tokens       = 0;
    int32_t     generated_tokens    = 0;
    int32_t     context_used_tokens = 0;
    int32_t     context_max_tokens  = 0;
    double      prefill_tok_per_sec = 0.0;
    double      decode_tok_per_sec  = 0.0;
    std::string error_message;
};

struct MemorySnapshot {
    double  process_rss_mib      = 0.0;
    double  process_peak_rss_mib = 0.0;
    double  process_swap_mib     = 0.0;
    double  system_total_mib     = 0.0;
    double  system_avail_mib     = 0.0;
    double  npu_htp_buffer_mib   = 0.0;
    double  host_weight_mib      = 0.0;
    int32_t context_used_tokens  = 0;
    int32_t context_max_tokens   = 0;
};

/**
 * Incremental UTF-8 byte assembler for byte-level BPE tokenizers (Llama 3, Qwen, Gemma, DeepSeek).
 * Ensures partial multi-byte UTF-8 sequences (such as 2-byte Cyrillic letters split across two tokens)
 * are buffered until complete, preventing JNI Modified-UTF8 crashes and garbled characters.
 */
class Utf8StreamAssembler {
public:
    std::string push(const char * bytes, size_t len);
    std::string flush();
    void clear();

private:
    std::string pending_;
};

class NpuEngine {
public:
    NpuEngine();
    ~NpuEngine();

    NpuEngine(const NpuEngine &) = delete;
    NpuEngine & operator=(const NpuEngine &) = delete;

    // Validates the 8-byte GGUF header ("GGUF" magic + format version 2 or 3)
    static bool validateGgufHeaderBytes(const uint8_t * data, size_t len, std::string * err_out = nullptr);

    InitResult init(
        const std::string & native_lib_dir,
        const std::string & htp_devices,
        const std::string & power_mode);

    ModelLoadResult loadModelFromFd(
        int detached_fd,
        const std::string & display_name,
        const EngineConfig & config,
        const std::function<void(float)> & on_progress = nullptr);

    ModelLoadResult loadModelFromPath(
        const std::string & file_path,
        const EngineConfig & config,
        const std::function<void(float)> & on_progress = nullptr);

    bool applyConfig(const EngineConfig & config, std::string * err_out = nullptr);

    void clearChat();
    void unloadModel();

    void requestCancel();

    int32_t countTokens(const std::string & text) const;

    GenerationStats generateReply(
        const std::vector<ChatTurnMessage> & messages,
        const EngineConfig & config,
        const std::function<bool(const std::string & piece)> & on_token_delta);

    MemorySnapshot readMemorySnapshot() const;

    bool isNpuReady() const { return npu_ready_.load(); }
    bool isModelLoaded() const { return model_loaded_.load(); }

    void onNativeLogLine(int level, const char * text);

private:
    ModelLoadResult finishModelLoadLocked(
        llama_model * loaded_model,
        const EngineConfig & sanitized_cfg);

    bool recreateContextLocked(const EngineConfig & cfg, std::string * err_out);
    void rebuildSamplerLocked(const EngineConfig & cfg);
    void unloadModelLocked();

    mutable std::mutex        lifecycle_mutex_;
    mutable std::shared_mutex vocab_mutex_;

    std::string native_lib_dir_;
    EngineConfig active_config_;

    std::vector<ggml_backend_device *> htp_devices_;
    std::vector<ggml_backend_device *> htp_devices_null_term_;
    std::string active_devices_label_;

    llama_model *       model_   = nullptr;
    const llama_vocab * vocab_   = nullptr;
    llama_context *     ctx_     = nullptr;
    llama_sampler *     sampler_ = nullptr;

    std::vector<int32_t> session_tokens_;

    std::atomic<bool>    backend_initialized_{false};
    std::atomic<bool>    npu_ready_{false};
    std::atomic<bool>    model_loaded_{false};
    std::atomic<bool>    cancel_requested_{false};

    std::atomic<int32_t> context_used_tokens_{0};
    std::atomic<int32_t> context_max_tokens_{0};
    std::atomic<double>  npu_htp_buffer_mib_{0.0};
    std::atomic<double>  host_weight_mib_{0.0};

    // Parsed asynchronously from llama/ggml-hex log stream
    mutable std::mutex log_parse_mutex_;
    std::string        log_line_accum_;
    int32_t            parsed_hexagon_arch_ = 0;
    int32_t            parsed_hvx_          = 0;
    int32_t            parsed_hmx_          = 0;
    int32_t            parsed_vtcm_mb_      = 0;
    int32_t            parsed_layers_npu_   = 0;
    int32_t            parsed_layers_total_ = 0;
    double             parsed_htp_mib_      = 0.0;
    double             parsed_cpu_mib_      = 0.0;
    std::string        last_hex_error_;
};

} // namespace npu

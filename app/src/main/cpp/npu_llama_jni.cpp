#include "npu_engine.hpp"

#include <jni.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

std::mutex g_instance_mutex;
std::unique_ptr<npu::NpuEngine> g_engine;

npu::NpuEngine & get_engine() {
    std::lock_guard<std::mutex> lk(g_instance_mutex);
    if (!g_engine) {
        g_engine = std::make_unique<npu::NpuEngine>();
    }
    return *g_engine;
}

/**
 * Converts standard UTF-8 (including 2-byte Cyrillic and 4-byte supplementary emojis)
 * into UTF-16 jchar units and constructs a jstring via NewString().
 * Avoids Android ART CheckJNI crashes caused by NewStringUTF() rejecting 4-byte UTF-8 emojis (0xF0..0xF4).
 */
jstring utf8_to_jstring(JNIEnv * env, const std::string & utf8) {
    if (!env) return nullptr;
    if (utf8.empty()) {
        const jchar empty = 0;
        return env->NewString(&empty, 0);
    }

    std::vector<jchar> utf16;
    utf16.reserve(utf8.size());

    const size_t n = utf8.size();
    size_t i = 0;
    while (i < n) {
        uint32_t cp = 0;
        const unsigned char b0 = static_cast<unsigned char>(utf8[i]);
        if ((b0 & 0x80u) == 0u) {
            cp = b0;
            i += 1;
        } else if ((b0 & 0xE0u) == 0xC0u && i + 1 < n) {
            const unsigned char b1 = static_cast<unsigned char>(utf8[i + 1]);
            cp = ((b0 & 0x1Fu) << 6) | (b1 & 0x3Fu);
            i += 2;
        } else if ((b0 & 0xF0u) == 0xE0u && i + 2 < n) {
            const unsigned char b1 = static_cast<unsigned char>(utf8[i + 1]);
            const unsigned char b2 = static_cast<unsigned char>(utf8[i + 2]);
            cp = ((b0 & 0x0Fu) << 12) | ((b1 & 0x3Fu) << 6) | (b2 & 0x3Fu);
            i += 3;
        } else if ((b0 & 0xF8u) == 0xF0u && i + 3 < n) {
            const unsigned char b1 = static_cast<unsigned char>(utf8[i + 1]);
            const unsigned char b2 = static_cast<unsigned char>(utf8[i + 2]);
            const unsigned char b3 = static_cast<unsigned char>(utf8[i + 3]);
            cp = ((b0 & 0x07u) << 18) | ((b1 & 0x3Fu) << 12) | ((b2 & 0x3Fu) << 6) | (b3 & 0x3Fu);
            i += 4;
        } else {
            cp = 0xFFFDu;
            i += 1;
        }

        if (cp <= 0xFFFFu) {
            utf16.push_back(static_cast<jchar>(cp));
        } else if (cp <= 0x10FFFFu) {
            cp -= 0x10000u;
            utf16.push_back(static_cast<jchar>(0xD800u | (cp >> 10)));
            utf16.push_back(static_cast<jchar>(0xDC00u | (cp & 0x3FFu)));
        } else {
            utf16.push_back(static_cast<jchar>(0xFFFDu));
        }
    }

    return env->NewString(utf16.data(), static_cast<jsize>(utf16.size()));
}

/**
 * Converts a Java UTF-16 jstring into standard UTF-8 (preserving 4-byte emojis and Cyrillic).
 */
std::string jstring_to_utf8(JNIEnv * env, jstring jstr) {
    if (!env || !jstr) {
        return {};
    }
    const jsize len = env->GetStringLength(jstr);
    if (len <= 0) {
        return {};
    }
    const jchar * chars = env->GetStringChars(jstr, nullptr);
    if (!chars) {
        return {};
    }

    std::string out;
    out.reserve(static_cast<size_t>(len) * 2);

    for (jsize i = 0; i < len; ++i) {
        uint32_t cp = chars[i];
        if (cp >= 0xD800u && cp <= 0xDBFFu && i + 1 < len) {
            const uint32_t low = chars[i + 1];
            if (low >= 0xDC00u && low <= 0xDFFFu) {
                cp = 0x10000u + (((cp - 0xD800u) << 10) | (low - 0xDC00u));
                ++i;
            }
        }

        if (cp <= 0x7Fu) {
            out.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FFu) {
            out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        } else if (cp <= 0xFFFFu) {
            out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        } else {
            out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        }
    }

    env->ReleaseStringChars(jstr, chars);
    return out;
}

npu::EngineConfig extract_config(
    JNIEnv * env,
    jint n_ctx,
    jint max_new_tokens,
    jint n_batch,
    jint n_threads,
    jint n_npu_layers,
    jfloat temperature,
    jfloat top_p,
    jint top_k,
    jfloat min_p,
    jfloat repeat_penalty,
    jint repeat_last_n,
    jint seed,
    jint flash_attn_mode,
    jboolean kv_q8,
    jstring power_mode,
    jstring htp_devices,
    jstring system_prompt) {
    npu::EngineConfig cfg;
    cfg.n_ctx           = n_ctx;
    cfg.max_new_tokens  = max_new_tokens;
    cfg.n_batch         = n_batch;
    cfg.n_threads       = n_threads;
    cfg.n_npu_layers    = n_npu_layers;
    cfg.temperature     = temperature;
    cfg.top_p           = top_p;
    cfg.top_k           = top_k;
    cfg.min_p           = min_p;
    cfg.repeat_penalty  = repeat_penalty;
    cfg.repeat_last_n   = repeat_last_n;
    cfg.seed            = seed;
    cfg.flash_attn_mode = flash_attn_mode;
    cfg.kv_q8           = (kv_q8 == JNI_TRUE);
    cfg.power_mode      = jstring_to_utf8(env, power_mode);
    cfg.htp_devices     = jstring_to_utf8(env, htp_devices);
    cfg.system_prompt   = jstring_to_utf8(env, system_prompt);
    cfg.sanitize();
    return cfg;
}

jobject make_model_load_result(JNIEnv * env, const npu::ModelLoadResult & r) {
    jclass cls = env->FindClass("ru/npu/llamachat/engine/NativeModelLoadResult");
    if (!cls) return nullptr;
    jmethodID ctor = env->GetMethodID(
        cls,
        "<init>",
        "(ZLjava/lang/String;Ljava/lang/String;JJIIIIDDLjava/lang/String;Ljava/lang/String;)V");
    if (!ctor) return nullptr;

    jstring j_desc  = utf8_to_jstring(env, r.model_desc);
    jstring j_quant = utf8_to_jstring(env, r.quant_name);
    jstring j_dev   = utf8_to_jstring(env, r.npu_device_used);
    jstring j_err   = utf8_to_jstring(env, r.error_message);

    jobject obj = env->NewObject(
        cls,
        ctor,
        r.ok ? JNI_TRUE : JNI_FALSE,
        j_desc,
        j_quant,
        static_cast<jlong>(r.model_bytes),
        static_cast<jlong>(r.model_params),
        static_cast<jint>(r.n_ctx_train),
        static_cast<jint>(r.n_ctx_active),
        static_cast<jint>(r.n_layers_total),
        static_cast<jint>(r.n_layers_npu),
        static_cast<jdouble>(r.htp_buffer_mib),
        static_cast<jdouble>(r.host_buffer_mib),
        j_dev,
        j_err);

    env->DeleteLocalRef(j_desc);
    env->DeleteLocalRef(j_quant);
    env->DeleteLocalRef(j_dev);
    env->DeleteLocalRef(j_err);
    env->DeleteLocalRef(cls);
    return obj;
}

} // namespace

extern "C" {

JNIEXPORT jobject JNICALL
Java_ru_npu_llamachat_engine_NpuLlamaBridge_nativeInitNpu(
    JNIEnv * env,
    jobject /*thiz*/,
    jstring j_native_lib_dir,
    jstring j_htp_devices,
    jstring j_power_mode) {

    const std::string lib_dir = jstring_to_utf8(env, j_native_lib_dir);
    const std::string devs    = jstring_to_utf8(env, j_htp_devices);
    const std::string pmode   = jstring_to_utf8(env, j_power_mode);

    const npu::InitResult res = get_engine().init(lib_dir, devs, pmode);

    jclass cls = env->FindClass("ru/npu/llamachat/engine/NativeInitResult");
    if (!cls) return nullptr;
    jmethodID ctor = env->GetMethodID(
        cls,
        "<init>",
        "(ZIIZLjava/lang/String;Ljava/lang/String;Ljava/lang/String;)V");
    if (!ctor) return nullptr;

    jstring j_active = utf8_to_jstring(env, res.active_devices);
    jstring j_hw     = utf8_to_jstring(env, res.hw_summary);
    jstring j_err    = utf8_to_jstring(env, res.error_message);

    jobject obj = env->NewObject(
        cls,
        ctor,
        res.ok ? JNI_TRUE : JNI_FALSE,
        static_cast<jint>(res.hexagon_arch),
        static_cast<jint>(res.htp_device_cnt),
        res.has_v73_skel ? JNI_TRUE : JNI_FALSE,
        j_active,
        j_hw,
        j_err);

    env->DeleteLocalRef(j_active);
    env->DeleteLocalRef(j_hw);
    env->DeleteLocalRef(j_err);
    env->DeleteLocalRef(cls);
    return obj;
}

JNIEXPORT jobject JNICALL
Java_ru_npu_llamachat_engine_NpuLlamaBridge_nativeLoadModelFromFd(
    JNIEnv * env,
    jobject /*thiz*/,
    jint detached_fd,
    jstring j_display_name,
    jint n_ctx,
    jint max_new_tokens,
    jint n_batch,
    jint n_threads,
    jint n_npu_layers,
    jfloat temperature,
    jfloat top_p,
    jint top_k,
    jfloat min_p,
    jfloat repeat_penalty,
    jint repeat_last_n,
    jint seed,
    jint flash_attn_mode,
    jboolean kv_q8,
    jstring j_power_mode,
    jstring j_htp_devices,
    jstring j_system_prompt,
    jobject j_progress_listener) {

    const std::string display_name = jstring_to_utf8(env, j_display_name);
    const npu::EngineConfig cfg = extract_config(
        env, n_ctx, max_new_tokens, n_batch, n_threads, n_npu_layers,
        temperature, top_p, top_k, min_p, repeat_penalty, repeat_last_n,
        seed, flash_attn_mode, kv_q8, j_power_mode, j_htp_devices, j_system_prompt);

    jmethodID on_prog_mid = nullptr;
    if (j_progress_listener) {
        jclass listener_cls = env->GetObjectClass(j_progress_listener);
        if (listener_cls) {
            on_prog_mid = env->GetMethodID(listener_cls, "onProgress", "(F)V");
            env->DeleteLocalRef(listener_cls);
        }
    }

    auto progress_cb = [&](float p) {
        if (j_progress_listener && on_prog_mid && !env->ExceptionCheck()) {
            env->CallVoidMethod(j_progress_listener, on_prog_mid, static_cast<jfloat>(p));
        }
    };

    const npu::ModelLoadResult res = get_engine().loadModelFromFd(
        static_cast<int>(detached_fd), display_name, cfg, progress_cb);
    return make_model_load_result(env, res);
}

JNIEXPORT jobject JNICALL
Java_ru_npu_llamachat_engine_NpuLlamaBridge_nativeLoadModelFromPath(
    JNIEnv * env,
    jobject /*thiz*/,
    jstring j_file_path,
    jint n_ctx,
    jint max_new_tokens,
    jint n_batch,
    jint n_threads,
    jint n_npu_layers,
    jfloat temperature,
    jfloat top_p,
    jint top_k,
    jfloat min_p,
    jfloat repeat_penalty,
    jint repeat_last_n,
    jint seed,
    jint flash_attn_mode,
    jboolean kv_q8,
    jstring j_power_mode,
    jstring j_htp_devices,
    jstring j_system_prompt,
    jobject j_progress_listener) {

    const std::string file_path = jstring_to_utf8(env, j_file_path);
    const npu::EngineConfig cfg = extract_config(
        env, n_ctx, max_new_tokens, n_batch, n_threads, n_npu_layers,
        temperature, top_p, top_k, min_p, repeat_penalty, repeat_last_n,
        seed, flash_attn_mode, kv_q8, j_power_mode, j_htp_devices, j_system_prompt);

    jmethodID on_prog_mid = nullptr;
    if (j_progress_listener) {
        jclass listener_cls = env->GetObjectClass(j_progress_listener);
        if (listener_cls) {
            on_prog_mid = env->GetMethodID(listener_cls, "onProgress", "(F)V");
            env->DeleteLocalRef(listener_cls);
        }
    }

    auto progress_cb = [&](float p) {
        if (j_progress_listener && on_prog_mid && !env->ExceptionCheck()) {
            env->CallVoidMethod(j_progress_listener, on_prog_mid, static_cast<jfloat>(p));
        }
    };

    const npu::ModelLoadResult res = get_engine().loadModelFromPath(file_path, cfg, progress_cb);
    return make_model_load_result(env, res);
}

JNIEXPORT jstring JNICALL
Java_ru_npu_llamachat_engine_NpuLlamaBridge_nativeApplyConfig(
    JNIEnv * env,
    jobject /*thiz*/,
    jint n_ctx,
    jint max_new_tokens,
    jint n_batch,
    jint n_threads,
    jint n_npu_layers,
    jfloat temperature,
    jfloat top_p,
    jint top_k,
    jfloat min_p,
    jfloat repeat_penalty,
    jint repeat_last_n,
    jint seed,
    jint flash_attn_mode,
    jboolean kv_q8,
    jstring j_power_mode,
    jstring j_htp_devices,
    jstring j_system_prompt) {

    const npu::EngineConfig cfg = extract_config(
        env, n_ctx, max_new_tokens, n_batch, n_threads, n_npu_layers,
        temperature, top_p, top_k, min_p, repeat_penalty, repeat_last_n,
        seed, flash_attn_mode, kv_q8, j_power_mode, j_htp_devices, j_system_prompt);

    std::string err;
    const bool ok = get_engine().applyConfig(cfg, &err);
    return utf8_to_jstring(env, ok ? "" : err);
}

JNIEXPORT jobject JNICALL
Java_ru_npu_llamachat_engine_NpuLlamaBridge_nativeGenerate(
    JNIEnv * env,
    jobject /*thiz*/,
    jobjectArray j_roles,
    jobjectArray j_contents,
    jint n_ctx,
    jint max_new_tokens,
    jint n_batch,
    jint n_threads,
    jint n_npu_layers,
    jfloat temperature,
    jfloat top_p,
    jint top_k,
    jfloat min_p,
    jfloat repeat_penalty,
    jint repeat_last_n,
    jint seed,
    jint flash_attn_mode,
    jboolean kv_q8,
    jstring j_power_mode,
    jstring j_htp_devices,
    jstring j_system_prompt,
    jobject j_token_listener) {

    std::vector<npu::ChatTurnMessage> messages;
    if (j_roles && j_contents) {
        const jsize n_r = env->GetArrayLength(j_roles);
        const jsize n_c = env->GetArrayLength(j_contents);
        const jsize count = (n_r < n_c) ? n_r : n_c;
        messages.reserve(static_cast<size_t>(count));

        for (jsize i = 0; i < count; ++i) {
            auto j_r = static_cast<jstring>(env->GetObjectArrayElement(j_roles, i));
            auto j_c = static_cast<jstring>(env->GetObjectArrayElement(j_contents, i));
            messages.push_back(npu::ChatTurnMessage{
                jstring_to_utf8(env, j_r),
                jstring_to_utf8(env, j_c)
            });
            if (j_r) env->DeleteLocalRef(j_r);
            if (j_c) env->DeleteLocalRef(j_c);
        }
    }

    const npu::EngineConfig cfg = extract_config(
        env, n_ctx, max_new_tokens, n_batch, n_threads, n_npu_layers,
        temperature, top_p, top_k, min_p, repeat_penalty, repeat_last_n,
        seed, flash_attn_mode, kv_q8, j_power_mode, j_htp_devices, j_system_prompt);

    jmethodID on_token_mid = nullptr;
    if (j_token_listener) {
        jclass listener_cls = env->GetObjectClass(j_token_listener);
        if (listener_cls) {
            on_token_mid = env->GetMethodID(listener_cls, "onTokenDelta", "(Ljava/lang/String;)Z");
            env->DeleteLocalRef(listener_cls);
        }
    }

    auto token_cb = [&](const std::string & delta) -> bool {
        if (!j_token_listener || !on_token_mid) {
            return true;
        }
        jstring j_piece = utf8_to_jstring(env, delta);
        if (!j_piece) {
            return false;
        }
        const jboolean cont = env->CallBooleanMethod(j_token_listener, on_token_mid, j_piece);
        // CRITICAL: Release local ref on every token to prevent 512-entry JNI local ref table overflow!
        env->DeleteLocalRef(j_piece);
        if (env->ExceptionCheck()) {
            return false;
        }
        return cont == JNI_TRUE;
    };

    const npu::GenerationStats stats = get_engine().generateReply(messages, cfg, token_cb);

    jclass cls = env->FindClass("ru/npu/llamachat/engine/NativeGenerationStats");
    if (!cls) return nullptr;
    jmethodID ctor = env->GetMethodID(
        cls,
        "<init>",
        "(ZZIIIIIDDLjava/lang/String;)V");
    if (!ctor) return nullptr;

    jstring j_err = utf8_to_jstring(env, stats.error_message);
    jobject obj = env->NewObject(
        cls,
        ctor,
        stats.ok ? JNI_TRUE : JNI_FALSE,
        stats.stopped_by_user ? JNI_TRUE : JNI_FALSE,
        static_cast<jint>(stats.prompt_tokens),
        static_cast<jint>(stats.cached_tokens),
        static_cast<jint>(stats.generated_tokens),
        static_cast<jint>(stats.context_used_tokens),
        static_cast<jint>(stats.context_max_tokens),
        static_cast<jdouble>(stats.prefill_tok_per_sec),
        static_cast<jdouble>(stats.decode_tok_per_sec),
        j_err);

    env->DeleteLocalRef(j_err);
    env->DeleteLocalRef(cls);
    return obj;
}

JNIEXPORT void JNICALL
Java_ru_npu_llamachat_engine_NpuLlamaBridge_nativeCancelGeneration(
    JNIEnv * /*env*/,
    jobject /*thiz*/) {
    get_engine().requestCancel();
}

JNIEXPORT void JNICALL
Java_ru_npu_llamachat_engine_NpuLlamaBridge_nativeClearChat(
    JNIEnv * /*env*/,
    jobject /*thiz*/) {
    get_engine().clearChat();
}

JNIEXPORT void JNICALL
Java_ru_npu_llamachat_engine_NpuLlamaBridge_nativeUnloadModel(
    JNIEnv * /*env*/,
    jobject /*thiz*/) {
    get_engine().unloadModel();
}

JNIEXPORT jint JNICALL
Java_ru_npu_llamachat_engine_NpuLlamaBridge_nativeCountTokens(
    JNIEnv * env,
    jobject /*thiz*/,
    jstring j_text) {
    const std::string text = jstring_to_utf8(env, j_text);
    return static_cast<jint>(get_engine().countTokens(text));
}

JNIEXPORT jobject JNICALL
Java_ru_npu_llamachat_engine_NpuLlamaBridge_nativeGetMemoryInfo(
    JNIEnv * env,
    jobject /*thiz*/) {
    const npu::MemorySnapshot m = get_engine().readMemorySnapshot();

    jclass cls = env->FindClass("ru/npu/llamachat/engine/NativeMemoryInfo");
    if (!cls) return nullptr;
    jmethodID ctor = env->GetMethodID(
        cls,
        "<init>",
        "(DDDDDDDII)V");
    if (!ctor) return nullptr;

    jobject obj = env->NewObject(
        cls,
        ctor,
        static_cast<jdouble>(m.process_rss_mib),
        static_cast<jdouble>(m.process_peak_rss_mib),
        static_cast<jdouble>(m.process_swap_mib),
        static_cast<jdouble>(m.system_total_mib),
        static_cast<jdouble>(m.system_avail_mib),
        static_cast<jdouble>(m.npu_htp_buffer_mib),
        static_cast<jdouble>(m.host_weight_mib),
        static_cast<jint>(m.context_used_tokens),
        static_cast<jint>(m.context_max_tokens));

    env->DeleteLocalRef(cls);
    return obj;
}

} // extern "C"

# Keep JNI bridge & native callback interfaces intact under R8 release shrinking
-keep class ru.npu.llamachat.engine.NpuLlamaBridge { *; }
-keep class ru.npu.llamachat.engine.NpuLlamaBridge$* { *; }
-keep class ru.npu.llamachat.engine.NativeInitResult { *; }
-keep class ru.npu.llamachat.engine.NativeModelLoadResult { *; }
-keep class ru.npu.llamachat.engine.NativeGenerationStats { *; }
-keep class ru.npu.llamachat.engine.NativeMemoryInfo { *; }
-keepclasseswithmembernames,includedescriptorclasses class * {
    native <methods>;
}

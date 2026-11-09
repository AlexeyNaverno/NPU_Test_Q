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

# UInt16TensorFactory вызывает закрытый JNI метод ORT 1.29 по имени.
-keepclassmembers class ai.onnxruntime.OnnxTensor {
    private static native long createTensorFromBuffer(long, long, java.nio.Buffer, int, long, long[], int);
}

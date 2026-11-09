package ru.npu.llamachat.image

/** Все файлы закреплены на конкретной ревизии; gitBlob означает SHA-1 Git blob вместо SHA-256 LFS. */
internal data class ModelFile(val path: String, val bytes: Long, val digest: String, val gitBlob: Boolean = false)
internal data class ImageModel(
    val id: String, val title: String, val repository: String, val revision: String,
    val folder: String, val files: List<ModelFile>, val canGenerate: Boolean
) {
    val totalBytes: Long get() = files.sumOf { it.bytes }
}

internal object ImageModelCatalog {
    val dreamShaper = ImageModel("dreamshaper", "DreamShaper 8 LCM — лёгкий", "latentdivergence/dreamshaper8-lcm-npu-8gen2",
        "6a2e524fe1e74f8050ea169f6cec1f725f956d90", "dreamshaper8-lcm-v73", listOf(
            ModelFile("text_encoder.onnx", 733, "fc06df873065d0129fcb1e42b85200486bc9414b6914f2f82ea69a8c8ed6c8b8"),
            ModelFile("text_encoder_qairt_context.bin", 178077696, "b079a8d6dbd8613010e1027bcc5c5dfa73e09e4e67dcf6aeb3c1f2b004cd9cca"),
            ModelFile("unet.onnx", 1714, "286a49715a74e7fc739ae972f27965b70458d62130b3f83e1c28d65eec30121b"),
            ModelFile("unet_qairt_context.bin", 882466816, "86285ff4e4cda4fbe69ea9d2997b308c5de4a960234b28d56ca5a4231500a316"),
            ModelFile("vae.onnx", 873, "a92718467028e6289be68e46e7000f5e99d0d14377ebfbec738cee8a05bef29d"),
            ModelFile("vae_qairt_context.bin", 67473408, "4aba13be2a85e892053cd9b52a3f5a12415f30d6ae9ec1e438ce593f2131e531"),
            ModelFile("tokenizer/vocab.json", 1059962, "469be27c5c010538f845f518c4f5e8574c78f7c8", true),
            ModelFile("tokenizer/merges.txt", 524619, "76e821f1b6f0a9709293c3b6b51ed90980b3166b", true)
        ), true)
    val anything = ImageModel("anything-v5", "Anything V5 INT8 — тяжёлый (пока без генерации)", "xZekki/anything-v5-int8-onnx",
        "43d5660428f9d40d34571b3ec191b35e0723efa0", "anything-v5-int8", listOf(
            ModelFile("text_encoder/model.onnx", 419223, "5e513044b9f677bd9adeba5f5dde60ec19d615cef7dfeb4a160a52371acb7c3e"),
            ModelFile("text_encoder/model_int8.onnx.data", 123424512, "834b5b73dacdadce6c147920451fd267fd62f3aa1d00ef2092c14648ddc35a66"),
            ModelFile("unet/model.onnx", 1891087, "1f6aa9f4f995de08ebd587a631acdaa2511bf49af29dd21683c775011c64d157"),
            ModelFile("unet/model_int8.onnx.data", 860852480, "718a9d24822adebf98eea087fd73d03d90cf29b444237119ff260123ba1355da"),
            ModelFile("vae_decoder/model.onnx", 198078197, "1f075dc6c5c5d55254a4e257a18ebe2b191ee2cf363178fc0a0d839c6d2b57fd"),
            ModelFile("vae_encoder/model.onnx", 136756675, "cff2250c3c62b1471e0687e4a2aa9005017113e4adfd0e210746854584125be3"),
            ModelFile("tokenizer/merges.txt", 573514, "1876098c4dcbf22af518537b3bd63102a05e1313", true),
            ModelFile("tokenizer/vocab.json", 1109372, "521b2c5def638d17b1e463ce25597140a1720005", true)
        ), false)
    val models = listOf(dreamShaper, anything)
}

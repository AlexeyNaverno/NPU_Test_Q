package ru.npu.llamachat.image

/** Параметры одного uint16-тензора QNN. Значения берутся из метаданных модели. */
data class AffineUInt16Quantizer(val scale: Double, val zeroPoint: Int) {
    init {
        require(scale.isFinite() && scale > 0.0) { "Масштаб должен быть конечным и положительным" }
        require(zeroPoint in 0..65535) { "Zero point вне диапазона uint16" }
    }

    /** ONNX QuantizeLinear: округление до ближайшего чётного, затем насыщение в uint16. */
    fun quantize(value: Float): Int {
        require(value.isFinite()) { "Тензор содержит NaN или бесконечность" }
        val rounded = Math.rint(value.toDouble() / scale) + zeroPoint
        return rounded.coerceIn(0.0, 65535.0).toInt()
    }

    fun dequantize(value: Int): Float {
        require(value in 0..65535) { "Значение вне диапазона uint16" }
        return ((value - zeroPoint) * scale).toFloat()
    }
}

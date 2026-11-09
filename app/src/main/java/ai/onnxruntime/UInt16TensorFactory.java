package ai.onnxruntime;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Временный адаптер к внутреннему JNI ORT 1.29.0: публичная Java-фабрика не создаёт UINT16.
 * Требует закреплённой версии ORT и правил R8; при изменении версии необходима перепроверка ABI.
 */
public final class UInt16TensorFactory {
  private UInt16TensorFactory() {}

  /**
   * Создаёт тензор из текущего диапазона прямого ByteBuffer в native byte order.
   * Буфер и его память должны оставаться неизменными до окончания session.run и close тензора.
   * Возвращённый тензор удерживает ссылку на буфер и должен быть закрыт вызывающим кодом.
   */
  public static OnnxTensor create(OrtEnvironment env, ByteBuffer buffer, long[] shape)
      throws OrtException {
    if (env == null || buffer == null || shape == null) throw new NullPointerException();
    if (!buffer.isDirect() || buffer.order() != ByteOrder.nativeOrder()) {
      throw new IllegalArgumentException("Нужен прямой ByteBuffer в native byte order");
    }
    if (shape.length > TensorInfo.MAX_DIMENSIONS) throw new IllegalArgumentException("Слишком много измерений");
    long count = 1;
    for (long dim : shape) {
      if (dim <= 0 || count > Integer.MAX_VALUE / 2L / dim) {
        throw new IllegalArgumentException("Недопустимая форма UINT16 тензора");
      }
      count *= dim;
    }
    if (buffer.remaining() != count * 2) throw new IllegalArgumentException("Размер буфера не соответствует форме");
    ByteBuffer view = buffer.slice().order(ByteOrder.nativeOrder());
    long[] copiedShape = shape.clone();
    TensorInfo info = new TensorInfo(copiedShape, OnnxJavaType.INT16,
        TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16);
    try {
      Method create = OnnxTensor.class.getDeclaredMethod("createTensorFromBuffer",
          long.class, long.class, Buffer.class, int.class, long.class, long[].class, int.class);
      create.setAccessible(true);
      long handle = (Long) create.invoke(null, OnnxRuntime.ortApiHandle, env.defaultAllocator.handle,
          view, 0, (long) view.remaining(), copiedShape,
          TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16.value);
      if (handle == 0) throw new IllegalStateException("ORT вернул пустой UINT16 тензор");
      return new OnnxTensor(handle, env.defaultAllocator.handle, info, view, false);
    } catch (InvocationTargetException e) {
      Throwable cause = e.getCause();
      if (cause instanceof OrtException) throw (OrtException) cause;
      if (cause instanceof RuntimeException) throw (RuntimeException) cause;
      if (cause instanceof Error) throw (Error) cause;
      throw new IllegalStateException("Не удалось создать UINT16 тензор", cause);
    } catch (ReflectiveOperationException | SecurityException e) {
      throw new IllegalStateException("Несовместимая версия Java API ONNX Runtime", e);
    }
  }
}

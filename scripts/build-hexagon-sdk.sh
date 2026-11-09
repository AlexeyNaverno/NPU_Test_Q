#!/usr/bin/env bash
set -euo pipefail

# Сборка чистого NPU-пакета llama.cpp с Qualcomm Hexagon SDK (HTP v73/v75/v79/v81/v85) из исходников
# без OpenCL GPU (GGML_HEXAGON=ON, GGML_OPENCL=OFF) с использованием официального Docker-тулчейна.
# Если этот скрипт не запускался, Gradle автоматически скачает официальный пребилд llama.cpp Snapdragon.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_TAG="${LLAMA_SNAPDRAGON_TAG:-b12211}"
LLAMA_SRC_DIR="${ROOT_DIR}/third_party/llama.cpp"
OUT_SDK_DIR="${ROOT_DIR}/third_party/llama-snapdragon-sdk"
TOOLCHAIN_IMAGE="ghcr.io/snapdragon-toolchain/arm64-android:v0.8"

echo "==> Подготовка исходников llama.cpp (${LLAMA_TAG})..."
mkdir -p "${ROOT_DIR}/third_party"
if [ ! -d "${LLAMA_SRC_DIR}/.git" ]; then
    git clone --depth 1 --branch "${LLAMA_TAG}" https://github.com/ggml-org/llama.cpp.git "${LLAMA_SRC_DIR}"
fi

echo "==> Компиляция libllama + libggml-hexagon + libggml-htp-v73..v85 в контейнере ${TOOLCHAIN_IMAGE}..."
docker run --rm \
    -u "$(id -u):$(id -g)" \
    -v "${LLAMA_SRC_DIR}:/workspace/llama.cpp" \
    -v "${OUT_SDK_DIR}:/workspace/out-sdk" \
    -w /workspace/llama.cpp \
    "${TOOLCHAIN_IMAGE}" \
    bash -lc '
        cp -f docs/backend/snapdragon/CMakeUserPresets.json .
        cmake --preset arm64-android-snapdragon-release -B build-npu \
            -DGGML_HEXAGON=ON \
            -DGGML_OPENCL=OFF \
            -DLLAMA_CURL=OFF \
            -DBUILD_SHARED_LIBS=ON
        cmake --build build-npu -j "$(nproc)"
        cmake --install build-npu --prefix /workspace/out-sdk
    '

echo "==> Готово! SDK Qualcomm Hexagon NPU установлен в: ${OUT_SDK_DIR}"
echo "==> Теперь выполните: ./gradlew assembleRelease"

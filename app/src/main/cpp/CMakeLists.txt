cmake_minimum_required(VERSION 3.22.1)
project(npu_llama_jni VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (ANDROID)
    if (NOT DEFINED LLAMA_SNAPDRAGON_SDK_DIR)
        message(FATAL_ERROR "LLAMA_SNAPDRAGON_SDK_DIR must be provided by Gradle (Snapdragon Hexagon NPU SDK).")
    endif()

    set(LLAMA_INC_DIR "${LLAMA_SNAPDRAGON_SDK_DIR}/include")
    set(LLAMA_LIB_DIR "${LLAMA_SNAPDRAGON_SDK_DIR}/jniLibs/${ANDROID_ABI}")

    if (NOT EXISTS "${LLAMA_INC_DIR}/llama.h")
        message(FATAL_ERROR "Missing llama.h in ${LLAMA_INC_DIR}. Run prepareSnapdragonNpuSdk first.")
    endif()
    if (NOT EXISTS "${LLAMA_LIB_DIR}/libggml-hexagon.so")
        message(FATAL_ERROR "Missing libggml-hexagon.so in ${LLAMA_LIB_DIR}. CPU/GPU-only builds are forbidden!")
    endif()
    if (NOT EXISTS "${LLAMA_LIB_DIR}/libggml-htp-v73.so")
        message(FATAL_ERROR "Missing libggml-htp-v73.so (Snapdragon 8s Gen 3 HTP v73 skel) in ${LLAMA_LIB_DIR}!")
    endif()
else()
    # Host verification build mode (Linux x86_64/arm64 test harness)
    if (NOT DEFINED LLAMA_HOST_PKG_DIR)
        message(FATAL_ERROR "LLAMA_HOST_PKG_DIR must be set for non-Android host test builds.")
    endif()
    set(LLAMA_INC_DIR "${LLAMA_HOST_PKG_DIR}/include")
    set(LLAMA_LIB_DIR "${LLAMA_HOST_PKG_DIR}/lib")
endif()

add_library(llama_prebuilt SHARED IMPORTED)
set_target_properties(llama_prebuilt PROPERTIES IMPORTED_LOCATION "${LLAMA_LIB_DIR}/libllama.so")

add_library(ggml_prebuilt SHARED IMPORTED)
set_target_properties(ggml_prebuilt PROPERTIES IMPORTED_LOCATION "${LLAMA_LIB_DIR}/libggml.so")

add_library(ggml_base_prebuilt SHARED IMPORTED)
set_target_properties(ggml_base_prebuilt PROPERTIES IMPORTED_LOCATION "${LLAMA_LIB_DIR}/libggml-base.so")

add_library(npu_llama_jni SHARED
    npu_engine.cpp
    npu_llama_jni.cpp
)

target_include_directories(npu_llama_jni PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${LLAMA_INC_DIR}"
)

target_compile_options(npu_llama_jni PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
)

if (ANDROID)
    find_library(ANDROID_LOG_LIB log REQUIRED)
    find_library(ANDROID_SYS_LIB android REQUIRED)

    target_link_options(npu_llama_jni PRIVATE "-Wl,--build-id=sha1")
    target_link_libraries(npu_llama_jni PRIVATE
        llama_prebuilt
        ggml_prebuilt
        ggml_base_prebuilt
        ${ANDROID_LOG_LIB}
        ${ANDROID_SYS_LIB}
        dl
    )
else()
    find_package(JNI QUIET)
    if (JNI_FOUND)
        target_include_directories(npu_llama_jni PRIVATE ${JNI_INCLUDE_DIRS})
    elseif (DEFINED HOST_JNI_STUB_DIR)
        target_include_directories(npu_llama_jni PRIVATE "${HOST_JNI_STUB_DIR}")
    endif()

    target_link_libraries(npu_llama_jni PUBLIC
        llama_prebuilt
        ggml_prebuilt
        ggml_base_prebuilt
        dl
        pthread
    )
endif()

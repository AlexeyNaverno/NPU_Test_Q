import java.net.URI
import java.nio.file.Files
import java.nio.file.StandardCopyOption

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.jetbrains.kotlin.android)
}

val appVersionName = "0.0.1"
val appVersionCode = 1

val llamaTag: String = providers.gradleProperty("llamaSnapdragonTag").orElse("b12211").get()
val customSdkProp: String? = providers.gradleProperty("llamaSnapdragonSdkDir").orNull
    ?: System.getenv("LLAMA_SNAPDRAGON_SDK_DIR")

val sdkCacheDir = rootProject.layout.projectDirectory.dir(".npu-sdk-cache")
val localBuiltSdkDir = rootProject.layout.projectDirectory.dir("third_party/llama-snapdragon-sdk")
val stagedSdkRoot = layout.buildDirectory.dir("snapdragon-npu-sdk")
val stagedIncludeDir = stagedSdkRoot.map { it.dir("include") }
val stagedJniLibsRootDir = stagedSdkRoot.map { it.dir("jniLibs") }

abstract class PrepareSnapdragonNpuSdkTask : DefaultTask() {
    @get:Input
    abstract val llamaReleaseTag: Property<String>

    @get:Input
    @get:Optional
    abstract val externalSdkPath: Property<String>

    @get:Internal
    abstract val cacheDir: DirectoryProperty

    @get:Internal
    abstract val localBuiltDir: DirectoryProperty

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @TaskAction
    fun prepare() {
        val outRoot = outputDir.get().asFile
        val outInclude = File(outRoot, "include")
        val outJniArm64 = File(outRoot, "jniLibs/arm64-v8a")
        outInclude.mkdirs()
        outJniArm64.mkdirs()

        val sourceRoot: File = resolveSourceSdkRoot()
        val srcInclude = findSubdir(sourceRoot, "include", "llama.h")
            ?: throw GradleException("Не найдены заголовочные файлы llama.h в SDK: ${sourceRoot.absolutePath}")
        val srcLib = findSubdir(sourceRoot, "lib", "libggml-hexagon.so")
            ?: findSubdir(sourceRoot, "bin", "libggml-hexagon.so")
            ?: throw GradleException(
                "КРИТИЧЕСКАЯ ОШИБКА NPU-СБОРКИ: В SDK (${sourceRoot.absolutePath}) отсутствует libggml-hexagon.so! " +
                    "Сборка без аппаратного бэкенда Qualcomm Hexagon NPU запрещена."
            )

        // Stage headers
        srcInclude.listFiles()?.filter { it.isFile && (it.name.endsWith(".h") || it.name.endsWith(".hpp")) }?.forEach { hdr ->
            Files.copy(hdr.toPath(), File(outInclude, hdr.name).toPath(), StandardCopyOption.REPLACE_EXISTING)
        }

        // Stage shared libraries (resolving any symlinks into concrete .so files for APK packaging)
        srcLib.listFiles()?.filter { it.name.endsWith(".so") }?.forEach { soFile ->
            val realPath = soFile.toPath().toRealPath()
            Files.copy(realPath, File(outJniArm64, soFile.name).toPath(), StandardCopyOption.REPLACE_EXISTING)
        }

        // Strictly verify mandatory Hexagon NPU + Snapdragon 8s Gen 3 (HTP v73) libraries
        val mandatoryLibs = listOf(
            "libllama.so",
            "libggml.so",
            "libggml-base.so",
            "libggml-cpu.so",
            "libggml-hexagon.so",
            "libggml-htp-v73.so" // Snapdragon 8s Gen 3 (SM8635) uses Hexagon HTP v73!
        )
        val missing = mandatoryLibs.filter { !File(outJniArm64, it).exists() }
        if (missing.isNotEmpty()) {
            throw GradleException(
                "КРИТИЧЕСКАЯ ОШИБКА NPU-СБОРКИ: отсутствуют обязательные библиотеки Hexagon NPU " +
                    "(включая libggml-htp-v73.so для Snapdragon 8s Gen 3): $missing"
            )
        }

        val stagedList = outJniArm64.listFiles()?.map { it.name }?.sorted() ?: emptyList()
        logger.lifecycle("Snapdragon Hexagon NPU SDK готов (${stagedList.size} .so): $stagedList")
    }

    private fun resolveSourceSdkRoot(): File {
        val custom = externalSdkPath.orNull?.trim().orEmpty()
        if (custom.isNotEmpty()) {
            val f = File(custom)
            if (!f.exists()) {
                throw GradleException("Указанный llamaSnapdragonSdkDir не существует: ${f.absolutePath}")
            }
            logger.lifecycle("Используется внешний Snapdragon NPU SDK: ${f.absolutePath}")
            return f
        }

        val localBuilt = localBuiltDir.get().asFile
        if (File(localBuilt, "lib/libggml-hexagon.so").exists()) {
            logger.lifecycle("Используется локально собранный Snapdragon NPU SDK: ${localBuilt.absolutePath}")
            return localBuilt
        }

        val tag = llamaReleaseTag.get()
        val cDir = cacheDir.get().asFile
        cDir.mkdirs()
        val archiveName = "llama-$tag-bin-android-arm64-snapdragon.tar.gz"
        val archiveFile = File(cDir, archiveName)
        val extractedDir = File(cDir, "extracted-$tag")

        if (!File(extractedDir, "llama-$tag/lib/libggml-hexagon.so").exists() &&
            !File(extractedDir, "lib/libggml-hexagon.so").exists()
        ) {
            if (!archiveFile.exists() || archiveFile.length() < 1_000_000L) {
                val url = "https://github.com/ggml-org/llama.cpp/releases/download/$tag/$archiveName"
                logger.lifecycle("Загрузка официального пакета llama.cpp Snapdragon Hexagon NPU ($tag): $url")
                val tmpFile = File(cDir, "$archiveName.part")
                URI(url).toURL().openStream().use { input ->
                    tmpFile.outputStream().use { output ->
                        input.copyTo(output, bufferSize = 256 * 1024)
                    }
                }
                Files.move(tmpFile.toPath(), archiveFile.toPath(), StandardCopyOption.REPLACE_EXISTING)
            }

            extractedDir.deleteRecursively()
            extractedDir.mkdirs()
            val proc = ProcessBuilder("tar", "-xzf", archiveFile.absolutePath, "-C", extractedDir.absolutePath)
                .redirectErrorStream(true)
                .start()
            val output = proc.inputStream.bufferedReader().readText()
            val code = proc.waitFor()
            if (code != 0) {
                throw GradleException("Ошибка распаковки $archiveName (код $code): $output")
            }
        }

        return if (File(extractedDir, "llama-$tag").isDirectory) {
            File(extractedDir, "llama-$tag")
        } else {
            extractedDir
        }
    }

    private fun findSubdir(root: File, subName: String, markerFile: String): File? {
        val direct = File(root, subName)
        if (File(direct, markerFile).exists()) return direct
        root.listFiles()?.filter { it.isDirectory }?.forEach { child ->
            val nested = File(child, subName)
            if (File(nested, markerFile).exists()) return nested
        }
        return null
    }
}

val prepareSnapdragonNpuSdk by tasks.registering(PrepareSnapdragonNpuSdkTask::class) {
    llamaReleaseTag.set(llamaTag)
    externalSdkPath.set(customSdkProp)
    cacheDir.set(sdkCacheDir)
    localBuiltDir.set(localBuiltSdkDir)
    outputDir.set(stagedSdkRoot)
}

android {
    namespace = "ru.npu.llamachat"
    compileSdk = 36
    ndkVersion = "28.2.13676358"

    defaultConfig {
        applicationId = "ru.npu.llamachat"
        // Android 12+ (API 31+) provides Build.SOC_MODEL / SOC_MANUFACTURER & modern Qualcomm FastRPC cDSP support
        minSdk = 31
        targetSdk = 36

        versionCode = appVersionCode
        versionName = appVersionName

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            // Strictly 64-bit ARMv8/ARMv9 for Qualcomm Snapdragon Hexagon NPU SoCs (8s Gen 3 = SM8635)
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++17", "-O3", "-fexceptions", "-frtti")
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DLLAMA_SNAPDRAGON_SDK_DIR=${stagedSdkRoot.get().asFile.absolutePath}"
                )
            }
        }
    }

    signingConfigs {
        create("releaseSigning") {
            val ksPath = System.getenv("RELEASE_STORE_FILE")
            if (!ksPath.isNullOrBlank() && File(ksPath).exists()) {
                storeFile = File(ksPath)
                storePassword = System.getenv("RELEASE_STORE_PASSWORD")
                keyAlias = System.getenv("RELEASE_KEY_ALIAS")
                keyPassword = System.getenv("RELEASE_KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
            versionNameSuffix = "-debug"
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            val relSign = signingConfigs.getByName("releaseSigning")
            signingConfig = if (relSign.storeFile != null) {
                relSign
            } else {
                signingConfigs.getByName("debug")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }

    sourceSets {
        getByName("main") {
            jniLibs.srcDir(stagedJniLibsRootDir)
        }
    }

    packaging {
        jniLibs {
            // CRITICAL FOR HEXAGON NPU:
            // FastRPC cDSP daemon requires libggml-htp-v73.so .. v85.so extracted as real files on disk
            // inside applicationInfo.nativeLibraryDir so cdsprpcd can load the DSP skel onto HTP0.
            useLegacyPackaging = true
            // CRITICAL: libggml-htp-*.so are Qualcomm QDSP6 Hexagon ELF binaries (EM_HEXAGON) containing
            // a custom SWIV ELF section for unsigned PD validation — NEVER run NDK llvm-strip on them!
            keepDebugSymbols += setOf("**/libggml-htp-*.so")
            pickFirsts += setOf("**/libc++_shared.so", "**/libomp.so")
        }
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    buildFeatures {
        viewBinding = true
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlin {
        jvmToolchain(17)
    }
}

tasks.named("preBuild") {
    dependsOn(prepareSnapdragonNpuSdk)
}

tasks.matching {
    it.name.contains("CMake", ignoreCase = true) ||
        it.name.contains("externalNativeBuild", ignoreCase = true) ||
        it.name.contains("JniLib", ignoreCase = true)
}.configureEach {
    dependsOn(prepareSnapdragonNpuSdk)
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.fragment.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.recyclerview)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.material)
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.kotlinx.coroutines.android)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}

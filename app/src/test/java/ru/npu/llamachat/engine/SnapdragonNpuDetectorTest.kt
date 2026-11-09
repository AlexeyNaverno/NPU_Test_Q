package ru.npu.llamachat.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SnapdragonNpuDetectorTest {

    @Test
    fun `recognizes Snapdragon 8s Gen 3 SM8635 as Hexagon HTP v73 with HVX-safe FP16 policy`() {
        val report = SnapdragonNpuDetector.evaluatePlatform(
            rawManufacturer = "Qualcomm",
            rawSocModel = "SM8635",
            hardware = "qcom",
            board = "cliff",
            isArm64 = true,
            hasCdspDriver = true
        )
        assertTrue(report.isQualcommPlatform)
        assertTrue(report.isSnapdragon8sGen3)
        assertEquals(73, report.expectedHexagonArch)
        assertFalse(report.supportsFp16Hmx)
        assertNull(report.blockingReason)
        assertTrue(report.marketingName.contains("Snapdragon 8s Gen 3"))
    }

    @Test
    fun `disables FP16 HMX on Snapdragon 7 Gen 4 SM7750 and unknown v73 boards`() {
        val sm7750 = SnapdragonNpuDetector.evaluatePlatform(
            rawManufacturer = "Qualcomm",
            rawSocModel = "SM7750",
            hardware = "qcom",
            board = "canoe",
            isArm64 = true,
            hasCdspDriver = true
        )
        assertTrue(sm7750.isQualcommPlatform)
        assertEquals(73, sm7750.expectedHexagonArch)
        assertFalse(sm7750.supportsFp16Hmx)
        assertNull(sm7750.blockingReason)

        val unknownQcom = SnapdragonNpuDetector.evaluatePlatform(
            rawManufacturer = "Qualcomm",
            rawSocModel = "SM7635",
            hardware = "qcom",
            board = "volcano",
            isArm64 = true,
            hasCdspDriver = true
        )
        assertTrue(unknownQcom.isQualcommPlatform)
        assertFalse(unknownQcom.supportsFp16Hmx)
    }

    @Test
    fun `recognizes compatible Snapdragon 8 Gen 3 and 8 Elite with FP16 HMX`() {
        val gen3 = SnapdragonNpuDetector.evaluatePlatform(
            rawManufacturer = "Qualcomm",
            rawSocModel = "SM8650",
            hardware = "qcom",
            board = "pineapple",
            isArm64 = true,
            hasCdspDriver = true
        )
        assertTrue(gen3.isQualcommPlatform)
        assertEquals(75, gen3.expectedHexagonArch)
        assertTrue(gen3.supportsFp16Hmx)
        assertNull(gen3.blockingReason)

        val elite = SnapdragonNpuDetector.evaluatePlatform(
            rawManufacturer = "QTI",
            rawSocModel = "SM8750",
            hardware = "qcom",
            board = "sun",
            isArm64 = true,
            hasCdspDriver = true
        )
        assertTrue(elite.isQualcommPlatform)
        assertEquals(79, elite.expectedHexagonArch)
        assertTrue(elite.supportsFp16Hmx)
        assertNull(elite.blockingReason)
    }

    @Test
    fun `strictly blocks non-Snapdragon SoCs and emulators`() {
        val mediatek = SnapdragonNpuDetector.evaluatePlatform(
            rawManufacturer = "MediaTek",
            rawSocModel = "MT6989",
            hardware = "mt6989",
            board = "k6989v1_64",
            isArm64 = true,
            hasCdspDriver = false
        )
        assertFalse(mediatek.isQualcommPlatform)
        assertNotNull(mediatek.blockingReason)

        val emulator = SnapdragonNpuDetector.evaluatePlatform(
            rawManufacturer = "Google",
            rawSocModel = "EMULATOR",
            hardware = "ranchu",
            board = "goldfish_arm64",
            isArm64 = true,
            hasCdspDriver = false
        )
        assertFalse(emulator.isQualcommPlatform)
        assertNotNull(emulator.blockingReason)

        val nonArm64 = SnapdragonNpuDetector.evaluatePlatform(
            rawManufacturer = "Qualcomm",
            rawSocModel = "SM8635",
            hardware = "qcom",
            board = "cliff",
            isArm64 = false,
            hasCdspDriver = true
        )
        assertNotNull(nonArm64.blockingReason)
    }
}

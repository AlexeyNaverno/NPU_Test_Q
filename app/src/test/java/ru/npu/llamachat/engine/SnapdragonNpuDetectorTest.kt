package ru.npu.llamachat.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SnapdragonNpuDetectorTest {

    @Test
    fun `recognizes Snapdragon 8s Gen 3 SM8635 as Hexagon HTP v73`() {
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
        assertNull(report.blockingReason)
        assertTrue(report.marketingName.contains("Snapdragon 8s Gen 3"))
    }

    @Test
    fun `recognizes compatible Snapdragon 8 Gen 3 and 8 Elite`() {
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

package ru.npu.llamachat.image

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class GenerationCrashMarkerTest {
    @Test fun reportsOnlyLastStageAndClearsMarker() {
        val marker = GenerationCrashMarker(InstrumentationRegistry.getInstrumentation().targetContext)
        marker.finish()
        marker.start()
        marker.stage("UNet: загрузка QNN")
        val report = marker.takePreviousExit()
        assertTrue(report.orEmpty().contains("UNet: загрузка QNN"))
        assertNull(marker.takePreviousExit())
    }
}

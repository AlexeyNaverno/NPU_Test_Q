package ru.npu.llamachat.image

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class ImageModelRecoveryTest {
    @get:Rule val temporary = TemporaryFolder()

    @Test fun `restores single interrupted backup without losing content`() {
        val model = ImageModelCatalog.dreamShaper
        val backup = File(temporary.root, ".${model.folder}-old-1")
        check(backup.mkdir())
        File(backup, "sentinel").writeText("old")
        ImageModelDownloader.recoverBackupIn(temporary.root, model)
        assertFalse(backup.exists())
        assertEquals("old", File(temporary.root, "${model.folder}/sentinel").readText())
    }

    @Test fun `multiple backups are never deleted or guessed`() {
        val model = ImageModelCatalog.anything
        val a = File(temporary.root, ".${model.folder}-old-a").apply { mkdir() }
        val b = File(temporary.root, ".${model.folder}-old-b").apply { mkdir() }
        assertThrows(IllegalArgumentException::class.java) { ImageModelDownloader.recoverBackupIn(temporary.root, model) }
        assertEquals(true, a.exists() && b.exists())
    }
}

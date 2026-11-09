package ru.npu.llamachat

import android.app.Application
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelStore
import androidx.lifecycle.ViewModelStoreOwner
import ru.npu.llamachat.ui.ChatViewModel
import ru.npu.llamachat.ui.ImageViewModel

/** Общая сессия UI и сервиса живёт в памяти процесса, а не в Activity. */
class NpuApplication : Application(), ViewModelStoreOwner {
    override val viewModelStore = ViewModelStore()
    val images: ImageViewModel by lazy {
        ViewModelProvider(this, ViewModelProvider.AndroidViewModelFactory.getInstance(this))[ImageViewModel::class.java]
    }
    val session: ChatViewModel by lazy {
        ViewModelProvider(this, ViewModelProvider.AndroidViewModelFactory.getInstance(this))[ChatViewModel::class.java]
    }
}

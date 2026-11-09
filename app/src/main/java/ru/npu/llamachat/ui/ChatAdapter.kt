package ru.npu.llamachat.ui

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import ru.npu.llamachat.databinding.ItemMessageAssistantBinding
import ru.npu.llamachat.databinding.ItemMessageUserBinding
import ru.npu.llamachat.model.ChatMessage
import ru.npu.llamachat.model.MessageRole
import java.util.Locale

class ChatAdapter : ListAdapter<ChatMessage, RecyclerView.ViewHolder>(DiffCallback) {

    override fun getItemViewType(position: Int): Int {
        return when (getItem(position).role) {
            MessageRole.USER -> VIEW_TYPE_USER
            MessageRole.ASSISTANT -> VIEW_TYPE_ASSISTANT
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val inflater = LayoutInflater.from(parent.context)
        return when (viewType) {
            VIEW_TYPE_USER -> UserViewHolder(
                ItemMessageUserBinding.inflate(inflater, parent, false)
            )
            else -> AssistantViewHolder(
                ItemMessageAssistantBinding.inflate(inflater, parent, false)
            )
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val item = getItem(position)
        when (holder) {
            is UserViewHolder -> holder.bind(item)
            is AssistantViewHolder -> holder.bind(item)
        }
    }

    class UserViewHolder(
        private val binding: ItemMessageUserBinding
    ) : RecyclerView.ViewHolder(binding.root) {
        fun bind(msg: ChatMessage) {
            binding.tvMessageContent.text = msg.content
            binding.tvMessageMeta.text = if (msg.tokenCount > 0) {
                "${msg.tokenCount} ток."
            } else {
                ""
            }
            binding.tvMessageMeta.visibility = if (msg.tokenCount > 0) View.VISIBLE else View.GONE
        }
    }

    class AssistantViewHolder(
        private val binding: ItemMessageAssistantBinding
    ) : RecyclerView.ViewHolder(binding.root) {
        fun bind(msg: ChatMessage) {
            binding.tvMessageContent.text = if (msg.isStreaming && msg.content.isEmpty()) {
                "Генерация на Hexagon NPU…"
            } else {
                msg.content
            }
            binding.progressStreaming.visibility = if (msg.isStreaming) View.VISIBLE else View.GONE

            val metaParts = mutableListOf<String>()
            metaParts.add("Hexagon NPU")
            if (msg.tokenCount > 0) {
                metaParts.add("${msg.tokenCount} ток.")
            }
            if (!msg.isStreaming && msg.tokPerSec > 0.05) {
                metaParts.add(String.format(Locale.US, "%.1f ток/с", msg.tokPerSec))
            }
            if (msg.wasStopped) {
                metaParts.add("остановлено")
            }
            binding.tvMessageMeta.text = metaParts.joinToString(" • ")
        }
    }

    private object DiffCallback : DiffUtil.ItemCallback<ChatMessage>() {
        override fun areItemsTheSame(oldItem: ChatMessage, newItem: ChatMessage): Boolean =
            oldItem.id == newItem.id

        override fun areContentsTheSame(oldItem: ChatMessage, newItem: ChatMessage): Boolean =
            oldItem == newItem
    }

    companion object {
        private const val VIEW_TYPE_USER = 1
        private const val VIEW_TYPE_ASSISTANT = 2
    }
}

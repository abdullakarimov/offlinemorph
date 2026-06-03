package com.offlinemorph.android.core.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo

data class TensorData(
    val name: String,
    val data: FloatArray,
    val shape: LongArray?,
)

object OrtValueUtils {
    fun extractFirstFloatArray(result: OrtSession.Result): FloatArray? {
        for (entry in result) {
            val floats = flattenOnnxValue(entry.value)
            if (floats != null && floats.isNotEmpty()) {
                return floats
            }
        }
        return null
    }

    fun extractFloatOutputs(result: OrtSession.Result): List<TensorData> {
        val outputs = mutableListOf<TensorData>()
        for (entry in result) {
            val tensor = entry.value ?: continue
            val floats = flattenAny(tensor.value) ?: continue
            val shape = (tensor.info as? TensorInfo)?.shape
            outputs += TensorData(
                name = entry.key,
                data = floats,
                shape = shape,
            )
        }
        return outputs
    }

    private fun flattenOnnxValue(value: OnnxValue): FloatArray? {
        if (value !is OnnxTensor) {
            return null
        }
        return flattenAny(value.value)
    }

    private fun flattenAny(value: Any?): FloatArray? {
        return when (value) {
            null -> null
            is FloatArray -> value
            is Array<*> -> {
                val parts = value.mapNotNull { flattenAny(it) }
                if (parts.isEmpty()) {
                    null
                } else {
                    val total = parts.sumOf { it.size }
                    val out = FloatArray(total)
                    var offset = 0
                    parts.forEach { part ->
                        part.copyInto(out, destinationOffset = offset)
                        offset += part.size
                    }
                    out
                }
            }
            else -> null
        }
    }
}

package com.offlinemorph.android.core.ml

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer

class OrtSessionFactory {
    private val environment: OrtEnvironment by lazy {
        OrtEnvironment.getEnvironment()
    }

    fun createSession(modelFile: File): OrtSession {
        require(modelFile.exists()) { "Model file does not exist: ${modelFile.absolutePath}" }
        val sessionOptions = OrtSession.SessionOptions()
        return environment.createSession(modelFile.absolutePath, sessionOptions)
    }

    fun createFloatTensor(data: FloatArray, shape: LongArray): OnnxTensor {
        return OnnxTensor.createTensor(environment, FloatBuffer.wrap(data), shape)
    }
}

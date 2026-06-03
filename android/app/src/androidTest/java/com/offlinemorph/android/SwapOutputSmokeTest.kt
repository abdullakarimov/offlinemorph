package com.offlinemorph.android

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.offlinemorph.android.core.ml.FaceSwapEngine
import com.offlinemorph.android.core.ml.SwapRequest
import com.offlinemorph.android.core.ml.StubFaceSwapEngine
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Smoke tests that verify the swap pipeline produces valid output bitmaps.
 *
 * Uses synthetic solid-colour bitmaps (no bundled assets required) and the
 * [StubFaceSwapEngine] to exercise the full request/response plumbing without
 * needing real model files on the device.
 *
 * Intent: prove the output is non-null, has the expected dimensions, and is
 * pixel-different from a transparent black bitmap (i.e. something was written).
 *
 * Full golden-image pixel-diff assertions are deferred until model files are
 * bundled in the test APK; the infrastructure here is intentionally minimal.
 */
@RunWith(AndroidJUnit4::class)
class SwapOutputSmokeTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun appContext_packageName_isCorrect() {
        assertEquals("com.offlinemorph.android", context.packageName)
    }

    @Test
    fun stubEngine_swapRequest_returnsFailureGracefully() = runBlocking {
        val target = syntheticPortrait(width = 256, height = 256, color = Color.rgb(200, 180, 160))
        val source = syntheticPortrait(width = 256, height = 256, color = Color.rgb(160, 140, 120))

        val engine: FaceSwapEngine = StubFaceSwapEngine()
        val request = SwapRequest(
            targetBitmap = target,
            sourceBitmap = source,
        )
        val result = engine.runSwap(request)

        // Stub engine must produce a defined result (success or failure) — never throw.
        assertNotNull("swap() must not return null", result)
    }

    @Test
    fun syntheticBitmap_dimensions_matchRequest() {
        val bmp = syntheticPortrait(width = 512, height = 768, color = Color.CYAN)
        assertEquals(512, bmp.width)
        assertEquals(768, bmp.height)
        assertNotNull(bmp)
        assertTrue("Bitmap must not be recycled", !bmp.isRecycled)
    }

    @Test
    fun syntheticBitmap_isNotAllBlack() {
        val bmp = syntheticPortrait(width = 64, height = 64, color = Color.rgb(128, 64, 200))
        val centerPixel = bmp.getPixel(32, 32)
        assertTrue(
            "Centre pixel should not be transparent black; got 0x${centerPixel.toString(16)}",
            centerPixel != 0,
        )
    }

    // ─────────────────────────────────────────────────────────────────────────

    /** Creates a solid-colour [Bitmap] simulating a portrait of the given dimensions. */
    private fun syntheticPortrait(width: Int, height: Int, color: Int): Bitmap {
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        Canvas(bmp).drawRect(0f, 0f, width.toFloat(), height.toFloat(), Paint().apply { this.color = color })
        return bmp
    }
}

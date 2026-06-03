package com.offlinemorph.android.core.ml

/**
 * Controls which faces in the target image or video frame are swapped.
 *
 * [SPECIFIC] — swap only the single face chosen by the thumbnail strip (index-based).
 * [ALL_FACES] — swap every detected face regardless of gender.
 * [MALE_ONLY] — swap only faces classified as male (requires genderage.onnx).
 * [FEMALE_ONLY] — swap only faces classified as female (requires genderage.onnx).
 */
enum class FaceFilterMode(val displayName: String) {
    SPECIFIC("Specific"),
    ALL_FACES("All"),
    MALE_ONLY("Male"),
    FEMALE_ONLY("Female"),
}

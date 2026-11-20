# OfflineMorph Android App

Android-specific implementation details for OfflineMorph.

For product vision, privacy guarantees, and cross-platform roadmap, see the repository root README: ../README.md

## Scope

This module contains the native Android application that runs portrait AI workflows fully on-device.

- Kotlin + Jetpack Compose UI and feature screens
- ONNX Runtime Android inference sessions
- OpenCV-based preprocessing and image-space transforms
- NNAPI-aware execution path with automatic fallback behavior
- Local image and video pipelines (decode, process, encode)
- Local model management, install, and discovery flows

## Implemented Android Stack

- Language and UI: Kotlin, Jetpack Compose, Material 3
- Build Toolchain: Android Gradle Plugin 8.5.2, Kotlin 1.9.24, Java 17
- Android Targets: minSdk 29, targetSdk 35, compileSdk 35
- ML Runtime: ONNX Runtime Android 1.18.0
- Vision and CV: OpenCV 4.10.0
- Async Runtime: Kotlin Coroutines 1.8.1
- Lifecycle and State: AndroidX Lifecycle Runtime + ViewModel Compose

## Prerequisites

- Android Studio (latest stable)
- JDK 17
- Android SDK 35 and platform tools
- An Android device or emulator running API 29+

## Build And Run

From this folder:

```bash
./gradlew :app:assembleDebug
./gradlew :app:installDebug
```

Build a release artifact:

```bash
./gradlew :app:assembleRelease
```

## Tests

Instrumented tests live in `app/src/androidTest/`. Model-backed tests run only when the models
are already installed in the app (download them from the Setup tab first) and are skipped
otherwise. Run them without uninstalling the app, because `connectedAndroidTest` wipes the
downloaded models:

```bash
./gradlew :app:installDebug :app:installDebugAndroidTest
adb shell am instrument -w com.offlinemorph.android.test/androidx.test.runner.AndroidJUnitRunner
```

`GoldenImageRegressionTest` compares outputs against reference PNGs in
`app/src/androidTest/assets/goldens/`. A missing golden is recorded on the device and the test is
skipped. To record or refresh goldens, run with `-e recordGoldens true` and pull them back:

```bash
adb pull /sdcard/Android/data/com.offlinemorph.android/files/goldens/. app/src/androidTest/assets/goldens/
```

## Release Signing

Release builds require signing configuration.

1. Copy keystore.properties.example to keystore.properties
2. Fill storeFile, storePassword, keyAlias, and keyPassword
3. Run the release build task

If release credentials are missing, the build intentionally fails fast.

## Model And Data Handling

- Inference runs locally on device with no cloud dependency.
- Model files are expected through local install/import flows.
- User images and video frames remain on-device during processing.

## Module Layout

```text
app/src/main/java/com/offlinemorph/android/
  core/
    image/      # bitmap loading and low-level image ops
    ml/         # ONNX sessions, tensor conversion, alignment, swap/enhance engines;
                # one subpackage per feature engine (aging, ancestry, beautify,
                # hairmakeup, relight, backgroundswap)
    video/      # decoder, encoder, frame indexing, video swap orchestration
  feature/
    aging/      # FRAN aging / de-aging screen
    ancestry/   # two-face identity blend screen
    beautify/   # skin smoothing and portrait retouch
    hairmakeup/ # face-parsing hair and lip colour
    flags/      # build-time feature flags
    consent/    # consent UX and policy gating
    device/     # capability assessment for local execution paths
    models/     # model catalog, download/install/import management
    swap/       # portrait swap screens and view models
    videoswap/  # video swap screens and view models
  ui/theme/     # Compose theme system
```

## Development Direction

- Continue native Android development as the primary mobile implementation track.
- Expand current ONNX + OpenCV on-device pipeline and quality controls.
- Improve thermal-aware scheduling and sustained performance on flagship NPUs.

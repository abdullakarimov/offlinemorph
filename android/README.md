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
    ml/         # ONNX sessions, tensor conversion, alignment, swap/enhance engines
    video/      # decoder, encoder, frame indexing, video swap orchestration
  feature/
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

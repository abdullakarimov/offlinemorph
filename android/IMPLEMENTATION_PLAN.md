# OfflineMorph Android Implementation Plan

## Objective

Implement the next generation of OfflineMorph features on Android while preserving the core product principles:

- 100% on-device AI execution
- no cloud media processing
- low-latency interactive experience
- stable performance on flagship-class hardware

## Scope

This plan covers Android delivery for the following compiled feature set:

1. Core Face Swap stabilization and quality uplift
2. AI Aging, De-Aging, and Ancestry synthesis
3. High-fidelity Hair and Makeup studio
4. Instant Beautifying and Virtual Plastic Surgery controls
5. Cinematic 3D Relighting
6. Micro-precision Background Swapping

## Current Baseline

The existing Android app already includes:

- native Kotlin + Jetpack Compose app structure
- ONNX Runtime Android-based inference path
- OpenCV-backed image processing utilities
- model management and local install/download flows
- image and video feature module structure

Relevant directories:

- `app/src/main/java/com/offlinemorph/android/core/ml/`
- `app/src/main/java/com/offlinemorph/android/core/video/`
- `app/src/main/java/com/offlinemorph/android/feature/`

## Hugging Face Candidate Model Map

The following Hugging Face model repositories were identified as candidate starting points for each planned feature.

Selection policy for integration:

1. Prefer ONNX-native repositories.
2. Prefer permissive licenses (MIT/Apache-2.0) for commercial readiness.
3. Validate model provenance, accuracy, and domain fit before shipping.
4. Benchmark each candidate on-device before adoption.

| Feature Area | Candidate Hugging Face Models | Integration Notes |
| --- | --- | --- |
| Core Face Swap | `ezioruan/inswapper_128.onnx`, `thebiglaskowski/inswapper_128.onnx`, `Kibawilson/inswapper_128.onnx` | Use as baseline swap generator candidates. Add license/provenance gate before production use. |
| Face Embeddings / Identity Backbone (supports swap + ancestry blend) | `onnx-community/arcface-onnx`, `garavv/arcface-onnx`, `immich-app/buffalo_l` | Evaluate embedding consistency and compatibility with existing analyzer pipeline. |
| Face Parsing (Hair/Lips/Brows/Skin) | `vivym/face-parsing-bisenet`, `Jasnk/face-parsing-bisenet-onnx` | Candidate semantic segmentation backbone for hair/makeup masks. |
| Portrait Enhancement (optional enhancer pass) | `Meeperomi/GFPGANv1.4-onnx`, `HowToSD/GFPGAN-ONNX`, `neurobytemind/GFPGANv1.4.onnx` | Candidate enhancer pass after swap/beautify output. Validate artifact behavior on mobile. |
| Hair-Level Background Matting | `onnx-community/BiRefNet-portrait-ONNX`, `onnx-community/BiRefNet_lite-ONNX`, `onnx-community/modnet-webnn`, `frankminors123/U2Net_Portrait_ONNX` | Start with `BiRefNet_lite` and `BiRefNet-portrait` for quality/perf comparison. |
| Relighting Depth Prior (for approximate normal estimation) | `onnx-community/depth-anything-v2-small`, `onnx-community/depth-anything-v2-base`, `Heliosoph/zoedepth-nyu-kitti-onnx` | Use monocular depth as first-stage prior for local relight engine. |
| Landmark Alternative (if not using MediaPipe Tasks) | `astaileyyoung/FaceMeshONNX` | Experimental fallback only; primary recommendation remains MediaPipe Tasks for production. |
| Aging / De-aging Generator | No strong ONNX mobile-ready match found via Hub query | Plan to export/convert a vetted aging model to ONNX and publish internal artifact mirror. |
| Ancestry Blend Generator | No strong ONNX mobile-ready match found via Hub query | Implement as internal latent-blend pipeline using ArcFace-style embeddings + custom generator. |

### Model Intake Checklist

Every candidate model must pass this checklist before entering the app build:

1. License review and legal approval.
2. Source/provenance verification.
3. ONNX graph validation with current `OrtSessionFactory` path.
4. INT8/FP16 quantization viability check.
5. Accuracy and artifact benchmark on target portrait set.
6. Thermal/memory benchmark on flagship Android hardware.
7. Safety review for misuse and unacceptable output behavior.

## Guiding Architecture Principles

1. Local-first always
- All feature pipelines must run fully on-device.
- No remote fallback is permitted for user media transforms.

2. Modular feature engines
- Each major feature gets a dedicated engine contract and implementation.
- Shared geometry, masks, embeddings, and session resources are reused.

3. Hardware-aware execution
- Prefer NPU-accelerated paths where available.
- Add graceful CPU fallback and quality scaling under thermal pressure.

4. Progressive model loading
- Keep required models minimal for first-run usability.
- Download/install optional feature packs on demand.

## Delivery Phases

### Phase A: Foundation Hardening

Goal: prepare shared infrastructure so new features can ship safely and consistently.

Work items:

1. Shared feature contracts
- Introduce request/result contract patterns for all new engines.
- Standardize cancellation, progress callbacks, and typed error envelopes.

2. Common face geometry service
- Single face detection + landmarks pass reused across features.
- Add cache for intermediate data to reduce repeat computation.

3. Expanded model manifest system
- Define required versus optional model tiers.
- Add checksum validation and model version metadata.

4. Execution policy manager
- Device-tier based quality policy.
- Thermal/memory guardrail policy and automatic downshift logic.

5. Local metrics instrumentation
- Track latency, memory peaks, and thermal throttling events locally.
- Ensure no media content is logged.

Exit criteria:

- unified engine API is available
- model tiering works in setup flow
- baseline benchmark harness reports latency and memory consistently

### Phase B: Core Face Swap Uplift (Existing Feature)

Goal: improve quality, reliability, and consistency of current face swap.

Work items:

1. Boundary and blend improvements
- Improve seam handling and skin tone continuity around merge edges.

2. Multi-face target handling
- Better target-face selection UX and confidence guidance.

3. Quality profiles
- Add Fast, Balanced, Studio quality profiles with deterministic behavior.

4. Video consistency
- Improve temporal smoothing for frame-to-frame output stability.

Exit criteria:

- swap quality passes visual regression suite on curated test set
- target latency meets profile-specific thresholds on flagship devices

### Phase C: AI Aging, De-Aging, and Ancestry

Goal: ship the first net-new synthesis feature family.

Work items:

1. Aging engine
- Introduce age progression/regression controls with identity retention.

2. Ancestry blend engine
- Accept two local face inputs and generate hybrid projection outputs.

3. UX controls
- Age slider, blend ratio slider, realism/intensity tuning.

4. Safety and consent
- Add clear in-app local-generation disclosure and consent gate.

Exit criteria:

- identity remains recognizable across age transformations
- ancestry blend outputs are stable and artifact-controlled
- interactive preview is responsive on target hardware tiers

### Phase D: Hair and Makeup Studio

Goal: deliver studio-style local try-on tools.

Work items:

1. Semantic face parsing
- Integrate dense segmentation for hair, lips, brows, and skin regions.

2. Virtual barber
- Hair color transformations (initial)
- style presets and texture-driven variants (iterative expansion)

3. Makeup engine
- Lipstick, eyeliner, and blush overlays driven by landmark regions.

4. Facial hair simulator
- Stubble, mustache, and beard overlays matched to chin/jaw contours.

5. GPU acceleration
- Shift compositing-heavy operations to shader-based paths.

Exit criteria:

- mask boundaries remain stable around hairline and lip edges
- controls update smoothly without full pipeline stalls

### Phase E: Beautifying and Virtual Plastic Surgery

Goal: provide controlled portrait enhancement and structural editing.

Work items:

1. Portrait enhancement stack
- blemish suppression
- teeth whitening
- skin tone evening
- wrinkle suppression

2. Structural remodeling controls
- jawline shaping
- eye widening
- nose contouring
- contour symmetry tuning

3. Spatial deformation safety
- TPS mesh warping with strict deformation caps and guardrails.

Exit criteria:

- edits remain natural under normal usage ranges
- no major geometric tearing around high-detail regions

### Phase F: Cinematic 3D Relighting

Goal: provide interactive local relighting with professional control feel.

Work items:

1. Surface normal estimation
- infer face normal maps from local inputs.

2. Light rig interaction model
- draggable virtual lights with intensity and softness controls.

3. GPU shading compositor
- apply highlight/shadow synthesis in real time.

Exit criteria:

- drag interactions remain smooth with minimal flicker
- shading coherence remains stable across short video sequences

### Phase G: Micro-Precision Background Swapping

Goal: deliver hair-strand-level portrait matting and robust background replacement.

Work items:

1. Hair-detail matting
- integrate high-detail portrait matting model with confidence output.

2. Compositing modes
- static image background replacement
- synthetic bokeh blur mode
- scene backdrop presets

3. Edge refinement and temporal stabilization
- anti-halo refinements
- temporal consistency for video output

Exit criteria:

- hair edges preserve fine strands with minimal haloing
- video matte stability holds under head motion

## Codebase Workstreams

### 1. Core ML Workstream

Add new engine modules under:

- `core/ml/` aging engine
- `core/ml/` ancestry engine
- `core/ml/` hair-makeup engine
- `core/ml/` beautify engine
- `core/ml/` relight engine
- `core/ml/` matting engine

Reuse and extend:

- `OrtSessionFactory.kt`
- `OrtValueUtils.kt`
- existing pre/postprocessing utilities

### 2. Feature and UI Workstream

Add new feature packages under:

- `feature/aging/`
- `feature/hairmakeup/`
- `feature/beautify/`
- `feature/relight/`
- `feature/matting/`

Follow existing state-driven ViewModel patterns used by current swap flow.

### 3. Models and Distribution Workstream

- expand model catalog entries and metadata
- add per-feature readiness indicators
- support optional pack installation without blocking base app use

### 4. Video Pipeline Workstream

- reuse decoder/encoder primitives in `core/video/`
- add temporal smoothing for matte, relight, and beautify outputs
- add frame-budget controls for sustained sessions

## Performance Strategy

1. Device tiers

- Tier 1 (flagship): full-resolution feature set
- Tier 2: reduced resolution or reduced effect complexity

2. Runtime quality profiles

- Fast: low latency, reduced refinement
- Balanced: default quality/performance
- Studio: highest quality where thermal budget allows

3. Thermal guardrails

- detect sustained heat pressure
- automatically lower quality level and model complexity

4. Memory guardrails

- cap concurrent model residency
- enforce staged load/unload policies by feature mode

## Quality and Validation Plan

1. Visual regression testing
- golden image suite for swap seams, skin consistency, haloing, warping, and relight artifacts.

2. Performance benchmarking
- latency per feature mode
- memory peak tracking
- thermal throttling incidence rate

3. Stability testing
- long-session stress tests for crashes, memory leaks, and degraded output.

4. Privacy verification
- explicit checks that no user media leaves device paths.

## Safety and Product Policy

1. Keep all processing local and explicit to users.
2. Add consent disclosures for synthetic face generation features.
3. Provide optional output watermarking/metadata marker policy.
4. Ensure no hidden analytics pipeline ingests media payloads.

## Milestones and Timeline (Suggested)

1. M1 (2-3 weeks)
- Foundation hardening
- face swap quality uplift baseline

2. M2 (3-4 weeks)
- aging/de-aging/ancestry beta

3. M3 (4-5 weeks)
- hair and makeup studio beta

4. M4 (3-4 weeks)
- beautify and structural remodeling

5. M5 (3-4 weeks)
- cinematic relighting plus background swap

6. M6 (2-3 weeks)
- optimization, QA hardening, release candidate

## Sprint 1 Recommended Backlog

1. ✅ Implement expanded model manifest and feature flags.
2. ✅ Build shared geometry/mask cache service.
3. ✅ Deliver aging feature vertical slice (engine + UI + benchmarks).
4. ✅ Add benchmark harness for latency/memory/thermal metrics.
5. Add golden-image regression tests for swap and aging outputs.

## Implementation Progress

### Iteration 1 — Phase A Foundation (completed 2026-06-03)

Files added / modified:

- `core/ml/EngineContracts.kt` — `EngineError`, `EngineResult<T>`, `EngineProgress`
- `core/ml/FaceGeometryCache.kt` — thread-safe LRU bitmap→analysis cache
- `core/metrics/MetricsRecorder.kt` — local-only latency / memory / thermal recorder
- `feature/device/ExecutionPolicyManager.kt` — live thermal + memory → `QualityProfile`
- `feature/models/ModelCatalog.kt` — added `ModelTier`, `version`, `sha256`, `featurePack` to `ModelSpec`
- `feature/flags/FeatureFlags.kt` — build-time gates for Phases B–G (all off by default)
- `core/ml/aging/AgingEngine.kt` — `AgingRequest / AgingResult / AgingEngine` interface
- `core/ml/aging/StubAgingEngine.kt` — graceful stub returning `ModelNotFound`

### Iteration 2 — Phase A Wiring (completed 2026-06-03)

Files modified:

- `core/ml/FaceSwapEngine.kt` — added `maxImageSizePx` field to `SwapRequest`
- `core/ml/OnnxFaceAnalyzer.kt` — injected `FaceGeometryCache`; cache-hit path skips all inference
- `feature/models/LocalModelManager.kt` — added SHA-256 integrity check; models with a declared `sha256` in `ModelSpec` are validated on every `getStatus()` call
- `feature/swap/SwapViewModel.kt` — instantiates `ExecutionPolicyManager`; gates `enhancerEnabled` and passes `maxImageSizePx` through `SwapRequest` on every `startSwap()`

### Iteration 3 — Aging UI Vertical Slice + Phase B Swap Uplift (completed 2026-06-03)

Files added:

- `feature/aging/AgingUiState.kt` — sealed `AgingUiState` (Idle / Loading / Success / Error / Unavailable) + `AgingScreenState`
- `feature/aging/AgingViewModel.kt` — wired to `StubAgingEngine` behind `FeatureFlags.agingEnabled`; handles source pick, age-offset clamp, intensity, and pipeline lifecycle
- `feature/aging/AgingScreen.kt` — Compose screen with age slider (−50…+50), intensity slider, run / clear controls, result card, unavailable state card

Files modified:

- `OfflineMorphApp.kt` — `TABS` list built dynamically; "Aging" tab injected when `FeatureFlags.agingEnabled`; tab routing switched to name-based `when`; "Open Setup" dialog uses `TABS.indexOf("Setup")` so index is resilient to tab injection
- `core/ml/FaceSwapEngine.kt` — added `SwapRequest.scaledTo(maxPx)` extension that downscales both bitmaps to the longest-edge cap before inference
- `core/ml/OnDeviceFaceSwapEngine.kt` — `runSwap` applies `scaledRequest = request.scaledTo(maxImageSizePx)` at entry; all bitmap references in the single-face and multi-face paths use `scaledRequest`; recursive multi-face calls pass `maxImageSizePx = 0` to skip redundant re-scaling

Sprint 1 status:

- ✅ Item 1: expanded model manifest and feature flags
- ✅ Item 2: geometry/mask cache
- ✅ Item 3: aging vertical slice (engine contract + stub + UI; on-device model pending)
- ✅ Item 4: benchmark harness (MetricsRecorder)
- Item 5: golden-image regression tests (pending)

### Iteration 4 — Aging Tab Live + Phase C Ancestry + Phase B Poisson Seam (completed 2026-06-03)

Files added:

- `core/ml/ancestry/AncestryEngine.kt` — `AncestryRequest / AncestryResult / AncestryEngine` interface
- `core/ml/ancestry/StubAncestryEngine.kt` — graceful stub returning `ModelNotFound`

Files modified:

- `feature/flags/FeatureFlags.kt` — `agingEnabled = true`; Aging tab is now visible in the running app
- `feature/aging/AgingScreen.kt` — added inline synthetic-generation disclosure card (on-device notice + no-upload guarantee) shown above controls; uses `MaterialTheme.colorScheme.secondaryContainer` for distinction; added `HorizontalDivider` separator
- `core/ml/FaceAlignmentOps.kt` — added `org.opencv.photo.Photo` import; added `poissonSeamBlend()` helper using `Photo.seamlessClone(NORMAL_CLONE)` with alpha-derived matte and moment-centred anchor; called from `compositeSwapBack()` as a post-pass that replaces the alpha seam with Poisson-blended colour/gradient continuity; falls back to plain alpha composite on any OpenCV error

Phase B swap quality status: Poisson seam blending is active for all affine-composite swap paths.

### Iteration 4b — Aging Tab Crash Fix (completed 2026-06-03)

Root cause: `AndroidViewModelFactory` uses reflection to find an `(Application)` constructor on `AndroidViewModel` subclasses. Kotlin default-parameter constructors generate a synthetic constructor, not the standard `(Application, ...)` overload, so the factory throws `NoSuchMethodException`. Fixed by removing `agingEngine` from the `AgingViewModel` constructor and instantiating it as a private property instead.

Files modified:

- `feature/aging/AgingViewModel.kt` — removed `agingEngine: AgingEngine = StubAgingEngine()` from constructor; added `private val agingEngine: AgingEngine = StubAgingEngine()` as class body property

### Iteration 5 — Hair & Makeup + Beautify Contracts + Stub Screens + Golden-Image Test Infra (completed 2026-06-03)

Files added:

- `core/ml/hairmakeup/HairMakeupEngine.kt` — `HairMakeupRequest / HairMakeupResult / HairMakeupEngine` interface (Phase D)
- `core/ml/hairmakeup/StubHairMakeupEngine.kt` — graceful stub returning `ModelNotFound`
- `core/ml/beautify/BeautifyEngine.kt` — `BeautifyRequest / BeautifyResult / BeautifyEngine` interface (Phase E)
- `core/ml/beautify/StubBeautifyEngine.kt` — graceful stub returning `ModelNotFound`
- `feature/hairmakeup/HairMakeupScreen.kt` — "Coming Soon" screen with on-device disclosure card
- `feature/beautify/BeautifyScreen.kt` — "Coming Soon" screen with on-device disclosure card
- `androidTest/java/com/offlinemorph/android/SwapOutputSmokeTest.kt` — golden-image test infra: 3 smoke tests covering package name, stub engine result contract, and synthetic bitmap validity

Files modified:

- `feature/flags/FeatureFlags.kt` — `hairMakeupEnabled = true`, `beautifyEnabled = true` (two new tabs visible immediately)
- `OfflineMorphApp.kt` — added `HairMakeupScreen` and `BeautifyScreen` imports; TABS buildList now adds "Hair & Makeup" and "Beautify" behind their flags; routing `when` block handles both new tab names
- `app/build.gradle.kts` — added `androidTestImplementation` for `androidx.test.ext:junit:1.2.1`, `androidx.test:runner:1.6.1`, `androidx.test:core:1.6.1`

App now shows 6 tabs: Photo Swap · Video Swap · Aging · Hair & Makeup · Beautify · Setup.

### Iteration 6 — Setup Download Overhaul (completed 2026-06-03)

Goal: make the Setup tab download exactly the right models with live per-file progress bars.

Files added:
- none

Files modified:

- `feature/models/AndroidModelDownloader.kt` — full rewrite:
  - New `FileDownloadProgress(fileName, fraction, fileIndex, totalFiles)` data class for byte-level granularity
  - `downloadMissingRequiredModels` now iterates **only** `ModelCatalog.requiredModels` (bug fix: old code iterated all models)
  - New `downloadAllModels` method covering `ModelCatalog.allModels` (required + optional)
  - Private `download()` shared implementation; `downloadFile()` emits byte-level progress via `connection.contentLengthLong` with an 8 KB buffer; fraction = `bytesRead/contentLength`, or `-1f` if server omits `Content-Length`; already-present files emit fraction = 1f and are skipped

- `feature/swap/SwapUiState.kt` — added `ModelItemState` sealed interface (Idle / Installed / Downloading(fraction) / Done / Failed(reason)), `ModelDownloadItem(spec, state)` data class, and `downloadItems: List<ModelDownloadItem>` field to `SwapScreenState`

- `feature/swap/SwapViewModel.kt` — major update:
  - `refreshModelStatus()` now builds `downloadItems` from the union of installed + missing models and updates `_uiState` (respects `isWorking` flag to avoid race during active download)
  - Old `downloadModels()` replaced by private `runDownload(requiredOnly)` helper; calls `downloadMissingRequiredModels` or `downloadAllModels` with both `onProgress` (string) and `onFileProgress` (byte-level) callbacks
  - New public `downloadModels()` calls `runDownload(requiredOnly=true)`
  - New `downloadAllModels()` calls `runDownload(requiredOnly=false)`
  - Private `updateItemProgress(fileName, fraction)` maps fraction values to `ModelItemState`: `<0 → Downloading(-1f)` (indeterminate), `0..1 → Downloading(fraction)`, `≥1 → Done`

- `feature/swap/AiSetupScreen.kt` — full redesign:
  - Replaces two text-only status cards with a `ModelDownloadList` card showing every model as a `ModelRow`
  - Each `ModelRow` shows status icon (CheckCircle / CloudDownload / ErrorOutline / RadioButtonUnchecked), file name, role label, and a right-aligned percentage or state text
  - `LinearProgressIndicator` appears below the row while `ModelItemState.Downloading`; indeterminate when `fraction < 0`
  - "Download Required Models" is now the primary `Button` (full width, CTA)
  - "Download All (incl. Optional)" is an `OutlinedButton`
  - "Import AI Files" and "Assess Device" retained as `OutlinedButton`s in a `FlowRow`
  - Removed the old "Download AI Files" and "Refresh AI Pack" buttons; kept "Refresh" as a concise `OutlinedButton`
  - `material-icons-extended` dependency added to `app/build.gradle.kts` for `ErrorOutline` and `RadioButtonUnchecked`

### Next: Iteration 7 — Aging hidden, Beautify (OpenCV) + Hair & Makeup (BiSeNet) implemented

**Completed:**

1. `feature/flags/FeatureFlags.kt` — `agingEnabled = false` (Aging tab hidden; no publicly available ONNX aging model found)
2. `feature/models/ModelCatalog.kt` — added `FACE_PARSING = "face_parsing.onnx"`, `featurePackModels` list with face_parsing entry (`https://huggingface.co/Jasnk/face-parsing-bisenet-onnx/resolve/main/face_parsing.onnx`), updated `allModels`
3. `core/ml/beautify/OnDeviceBeautifyEngine.kt` — implements `BeautifyEngine` using OpenCV bilateral filter; no ONNX model required; `skinSmoothing` maps to bilateral sigma; blended with `Core.addWeighted`
4. `core/ml/hairmakeup/OnDeviceHairMakeupEngine.kt` — implements `HairMakeupEngine` using BiSeNet face-parsing ONNX; returns `EngineError.ModelNotFound` when `face_parsing.onnx` absent; argmax segmentation; per-class colour overlay for hair (class 17), lips (12+13), eyeshadow (2+3+4+5)
5. `feature/beautify/BeautifyUiState.kt` + `BeautifyViewModel.kt` — full state machine, plain `(Application)` constructor, `OnDeviceBeautifyEngine` as class-body property
6. `feature/hairmakeup/HairMakeupUiState.kt` + `HairMakeupViewModel.kt` — full state machine, `ModelNotReady` state wired from `EngineError.ModelNotFound`
7. `feature/beautify/BeautifyScreen.kt` — full rewrite: portrait picker, skin-smoothing/eye-enlarge/face-slim/teeth-whiten sliders, result card
8. `feature/hairmakeup/HairMakeupScreen.kt` — full rewrite: portrait picker, 11 hair colour chips, 8 lip colour chips, intensity slider, `ModelNotReady` card pointing user to Setup download
9. `OfflineMorphApp.kt` — added `HairMakeupViewModel` + `BeautifyViewModel` creation; both screens receive `viewModel` param

**Model availability policy (established):** Features without publicly available ONNX models are hidden via `FeatureFlags`; only features with a downloadable public model URL are enabled.

### Next: Iteration 8 — Ancestry UI Vertical Slice + Relight/Background Engine Contracts

Planned work:

1. `feature/ancestry/AncestryUiState.kt` — state sealed types (Idle/Loading/Success/Error/Unavailable)
2. `feature/ancestry/AncestryViewModel.kt` — AndroidViewModel wiring `StubAncestryEngine`; same constructor-body pattern used for `AgingViewModel`
3. `feature/ancestry/AncestryScreen.kt` — two-portrait picker, blend ratio slider (0..1), Run/Clear, pipeline card
4. `feature/flags/FeatureFlags.kt` — `ancestryEnabled = true`
5. Wire "Ancestry" tab into `OfflineMorphApp`
6. `core/ml/relight/RelightEngine.kt` + stub (Phase F)
7. `core/ml/backgroundswap/BackgroundSwapEngine.kt` + stub (Phase G)




A feature is considered done when all conditions are met:

1. It passes visual quality benchmarks on representative portrait sets.
2. It meets latency and memory targets on flagship baseline devices.
3. It degrades gracefully under thermal pressure.
4. It is fully functional offline without network dependency.
5. It has consent, safety copy, and QA test coverage in place.

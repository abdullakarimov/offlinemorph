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

### Next: Iteration 4 — Phase B Swap Quality Uplift + Phase C Ancestry Contract

Planned work:

1. `core/ml/ancestry/AncestryEngine.kt` — `AncestryRequest / AncestryResult / AncestryEngine` interface and stub
2. `feature/aging/` — add `AgingConsentCard` inline disclosure for synthetic-face generation (Phase C safety)
3. Phase B swap quality: `OnDeviceFaceSwapEngine` — add Poisson-blending seam pass after composite to reduce hard edges around the face boundary
4. Phase B swap quality: `OnnxFaceAnalyzer` — multi-face sorted output reliability improvements (NMS confidence threshold tuning)
5. Item 5: `androidTest/` — golden-image smoke test for swap output (basic bitmap diff infrastructure)



A feature is considered done when all conditions are met:

1. It passes visual quality benchmarks on representative portrait sets.
2. It meets latency and memory targets on flagship baseline devices.
3. It degrades gracefully under thermal pressure.
4. It is fully functional offline without network dependency.
5. It has consent, safety copy, and QA test coverage in place.

# Milestone 4 — Hardware-Aware Profiling Report

**Project:** MSML/MSAI 605 — Face Verification
**Model:** InceptionResnetV1 (FaceNet), pretrained on VGGFace2
**Date:** May 2, 2026
**Authors:** Rahul, Aksshaj

---

## 1. Measurement Environment

| Item | Value |
|------|-------|
| OS / Platform | macOS-15.6.1-arm64-arm-64bit |
| Python version | 3.11.14 |
| Processor | arm |
| Logical CPUs | 8 |
| RAM (total) | 16.0 GB |
| PyTorch version | 2.2.2 |
| CUDA available | False |
| Device used | CPU (required baseline) |

All measurements were taken on CPU. No GPU was used for this profiling run.

---

## 2. Pipeline Overview

Each face verification call consists of three sequential stages:

1. **Preprocessing (MTCNN)** — Load image → MTCNN face detection → crop and resize to 160×160 px. Applied to both images in the pair.
2. **Embedding (FaceNet)** — Run InceptionResnetV1 on each 160×160 face tensor → produce 512-dimensional L2-normalized embedding vector. Applied to both images.
3. **Scoring (Cosine)** — Compute cosine similarity between the two 512-d vectors. Return a score ∈ [−1, 1]. Compare against threshold 0.547739 to make a decision.

---

## 3. Methodology

- **Pairs used:** First 20 pairs from `outputs/pairs/test_pairs.csv`
- **Warmup:** 3 pairs processed before timing begins (model and OS caches warm)
- **Timer:** Python `time.perf_counter()` with nanosecond resolution
- **Per-stage timing:** Each stage timed independently per pair. Preprocessing includes both images; embedding includes both images; scoring is the single cosine computation.
- **Batch-size sensitivity:** Sequentially process N pairs (N from 1 to 50), measure wall-clock time, derive per-pair and throughput figures.
- **Repetitions:** Each batch size measured once (model is deterministic on CPU; variance comes from OS scheduling).

---

## 4. CPU Baseline — Per-Stage Latency

*n = 20 pairs, warmup = 3 pairs, device = CPU*

| Stage | Mean (ms) | Std (ms) | p50 (ms) | p95 (ms) | Min (ms) | Max (ms) |
|-------|----------:|--------:|--------:|--------:|--------:|--------:|
| Preprocessing (MTCNN) | 48.643 | 8.198 | 46.594 | 61.306 | 34.814 | 68.173 |
| Embedding (FaceNet) | 36.839 | 3.184 | 35.842 | 42.190 | 33.057 | 44.052 |
| Scoring (cosine) | 0.049 | 0.003 | 0.048 | 0.053 | 0.045 | 0.061 |
| **End-to-end** | **85.531** | **9.350** | **85.010** | **100.054** | **69.347** | **109.965** |

**Note:** Preprocessing and embedding each cover both images in the pair; scoring is a single vector operation.

---

## 5. Batch-Size Sensitivity

Batch size here refers to the number of verification pairs processed sequentially in a single call. The system currently processes pairs one by one (no batching of images through the neural network).

| Batch Size | Total Time (ms) | Per-Pair (ms) | Throughput (pairs/s) |
|----------:|---------------:|-------------:|--------------------:|
| 1 | 81.69 | 81.69 | 12.241 |
| 5 | 398.48 | 79.70 | 12.548 |
| 10 | 806.85 | 80.69 | 12.394 |
| 20 | 1575.99 | 78.80 | 12.690 |
| 50 | 4270.05 | 85.40 | 11.709 |

---

## 6. Interpretation

**Dominant stage:** Preprocessing (MTCNN face detection) actually dominates latency, contributing approximately **57%** of end-to-end time (48.643 ms / 85.531 ms). Embedding (FaceNet model inference) is the second-largest contributor at approximately **43%** of end-to-end time (36.839 ms / 85.531 ms). This is somewhat unusual — typically the deep CNN dominates — but on Apple Silicon CPU, MTCNN's multi-stage cascade (P-Net, R-Net, O-Net) applied to two full-resolution images outweighs a single forward pass through InceptionResnetV1 on already-cropped 160×160 inputs.

**Preprocessing (MTCNN):** Contributes approximately **57%** of end-to-end latency and shows the highest variance (std = 8.198 ms, range 34.814–68.173 ms). Variability comes from input image size and the number of detection candidates the cascade must evaluate. The fallback (centre-resize) is faster but less accurate.

**Embedding (FaceNet):** Contributes approximately **43%** of end-to-end latency with much tighter variance (std = 3.184 ms). The forward pass through InceptionResnetV1 on a fixed 160×160 input is highly consistent.

**Scoring:** Cosine similarity between two 512-d vectors is negligible at **0.049 ms** (~0.06% of end-to-end time), confirming that neural-network stages are the bottleneck, not the scoring step.

**Batch-size sensitivity:** Per-pair latency remains roughly constant across batch sizes (78.80–85.40 ms) because pairs are processed sequentially without image-level batching. Total time scales approximately linearly with batch size. Throughput is therefore approximately constant at **~12.3 pairs/second**. The slight degradation at batch size 50 (85.40 ms/pair, 11.709 pairs/s) is likely due to memory pressure or thermal throttling over longer runs.

**Practical implication:** For real-time 1-to-1 verification, the **~85.5 ms** end-to-end latency on CPU may be acceptable for non-interactive workflows (≈12 verifications/second). For lower latency, GPU inference, ONNX export, or replacing MTCNN with a lighter detector would be recommended — since MTCNN is the dominant stage, optimizing it would yield the largest gains.

**Load test reference (from Milestone 3):** Under 4-worker parallel load with 20 pairs, mean latency was 319.47 ms (p95 = 584.88 ms), throughput = 5.64 pairs/s. The higher per-request latency under parallel load is consistent with sequential CPU profiling: 4 workers contend for the same 8 logical cores, and MTCNN/FaceNet are already multi-threaded internally, so workers compete for CPU resources rather than scaling linearly.

---

## 7. Reproducibility

To reproduce this profiling report:
```bash
# From a clean clone with data already present
python scripts/profile_latency.py \
  --config configs/m3.yaml \
  --n-pairs 20 \
  --warmup 3 \
  --batch-sizes 1,5,10,20,50 \
  --output outputs/profiling_results.json
```
Results are saved to `outputs/profiling_results.json`. See also: `reports/reproducibility_checklist.md`.

*Final system version: tag `v1.0-final`. Config: `configs/m3.yaml`, threshold = 0.547739.*

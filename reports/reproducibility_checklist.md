# Reproducibility Checklist — MSML/MSAI 605 Milestone 4

**Project:** Face Verification System
**Release tag:** `v1.0-final`
**Authors:** Rahul, Aksshaj
**Config:** `configs/m3.yaml`
**Operating threshold:** 0.547739
---
## Prerequisites
- Python 3.11 installed
- Git installed
- Docker installed (for Option B)
- Internet access (first run downloads LFW dataset via TensorFlow Datasets)
---
## Step 1 — Clone the repository at the final tag
```bash
git clone https://github.com/rahulcs6104/msml605-face-ID.git 
cd msml605-face-ID
git checkout v1.0-final
```

---
## Step 2 — Set up the Python environment
**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
**Windows (PowerShell):**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Verify:
```bash
pip list | grep -E "facenet|torch|numpy|yaml"
```
---

## Step 3 — Ingest the LFW dataset and create pairs
```bash
python scripts/ingest_dataset.py --config configs/m1.yaml
python scripts/create_pairs.py   --config configs/m1.yaml
```
Expected outputs:
- `outputs/manifest.json`
- `outputs/splits.json`
- `outputs/pairs/train_pairs.csv`
- `outputs/pairs/val_pairs.csv`
- `outputs/pairs/test_pairs.csv`
- `data/lfw_images//.jpg`

---

## Step 4 — Re-calibrate the embedding threshold

```bash
python scripts/recalibrate.py --config configs/m3.yaml
```

Expected output:
- Console: `auto-updated inference.threshold in configs/m3.yaml → 0.547739`
- File: `configs/m3_threshold.json` updated
- File: `configs/m3.yaml` inference.threshold updated automatically

Verify alignment:
```bash
python -c "
import yaml, json
cfg = yaml.safe_load(open('configs/m3.yaml'))
thr = json.load(open('configs/m3_threshold.json'))
assert cfg['inference']['threshold'] == thr['threshold'], 'MISMATCH'
print('OK: threshold =', cfg['inference']['threshold'])
"
```

---

## Step 5 — Run M3 test evaluation (FaceNet metrics)

```bash
python scripts/evaluate_m3_test.py --config configs/m3.yaml
```

Expected output: `outputs/m3_test_metrics.json`
Key metric to verify: `threshold_used` should be 0.547739

---

## Step 6 — Run latency profiling

```bash
python scripts/profile_latency.py \
  --config configs/m3.yaml \
  --n-pairs 20 \
  --warmup 3 \
  --batch-sizes 1,5,10,20,50 \
  --output outputs/profiling_results.json
```

Expected output: `outputs/profiling_results.json`

---

## Step 7 — Run single-pair CLI inference (Option A: local)

```bash
python scripts/cli.py \
  --image-a data/lfw_images/Aaron_Eckhart/0000.jpg \
  --image-b data/lfw_images/Aaron_Eckhart/0000.jpg
```

Expected: Decision = "same", Score ≥ 0.547739, Confidence > 0.9

---

## Step 8 — Run batch CLI inference

```bash
python scripts/cli.py \
  --pairs-csv outputs/pairs/test_pairs.csv \
  --max-pairs 10
```

Expected: 10 results with decisions, scores, and latency breakdown printed.

---

## Step 9 — Run all tests

```bash
PYTHONPATH=. pytest tests/ -v
```

Expected: All tests pass (no failures).

---

## Step 10 — Run Dockerized CLI (Option B: Docker)
```bash
# Build the Docker image
docker build -t face-verify .

# Single-pair inference
docker run --rm -v $(pwd)/data:/app/data face-verify \
  --image-a data/lfw_images/Aaron_Eckhart/0000.jpg \
  --image-b data/lfw_images/Aaron_Eckhart/0000.jpg

# Batch inference
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  face-verify \
  --pairs-csv outputs/pairs/test_pairs.csv \
  --max-pairs 10
```

Expected: Same output as Step 7/8 but running inside Docker.

---

## Artifact Locations

| Artifact | Path in repo |
|----------|-------------|
| System Card PDF | `reports/system_card_m4.pdf` |
| Profiling report | `reports/profiling_report.md` |
| Reproducibility checklist | `reports/reproducibility_checklist.md` |
| Profiling results JSON | `outputs/profiling_results.json` |
| M3 test metrics JSON | `outputs/m3_test_metrics.json` |
| Final config | `configs/m3.yaml` |
| Threshold cache | `configs/m3_threshold.json` |
| Dockerfile | `Dockerfile` |
| CLI entrypoint | `scripts/cli.py` |

---

## Final Release

```bash
git tag v1.0-final
git push origin v1.0-final
```

Verify tag exists:
```bash
git show v1.0-final --stat
```
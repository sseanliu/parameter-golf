# Parameter Golf Autoresearch

Autonomous research loop for the OpenAI Parameter Golf challenge.

## Challenge Constraints

- **Artifact limit**: code_bytes + compressed_model_bytes <= 16,000,000 bytes
- **Training time**: 5 minutes per experiment (we use 5 min for fast iteration; final submission uses 10 min on 8xH100)
- **Hardware**: 1xH100 SXM 80GB (iteration); 8xH100 SXM (final)
- **Dataset**: Fixed FineWeb with SentencePiece 1024-token vocab
- **Metric**: val_bpb (bits per byte), lower is better
- **Current SOTA**: 1.1748 val_bpb (10 layers, 512 dim, int8+zlib)

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar19`).
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**:
   - This file (`program.md`) -- experiment instructions
   - `prepare.py` -- fixed constants, data loading, evaluation, quantization. **Do not modify.**
   - `train.py` -- the file you modify. Architecture, optimizer, hyperparameters.
4. **Verify data exists**: Check that `./data/datasets/fineweb10B_sp1024/` contains data shards. If not, run `python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`.
5. **Initialize results.tsv**: Create with header row only.
6. **Confirm and go**.

## Experimentation

Each experiment runs on a single H100 GPU. Training runs for a **fixed 5-minute wall clock budget**. Launch: `python train.py > run.log 2>&1`

**What you CAN modify:**
- `train.py` -- everything is fair game: architecture, optimizer, hyperparameters, quantization, depth, width, activation functions, attention patterns, initialization, etc.

**What you CANNOT modify:**
- `prepare.py` -- contains fixed evaluation, data loading, quantization, and size checking.
- Do not install new packages beyond what's in requirements.txt.

**The goal: get the lowest val_bpb while staying under the 16MB artifact limit.**

Key metrics printed at the end of each run:
```
val_bpb: X.XXXX
artifact_bytes: XXXXXXX (must be <= 16000000)
compressed_model_bytes: XXXXXXX
code_bytes: XXXXXXX
```

If artifact_bytes > 16,000,000, the run is INVALID regardless of val_bpb. You must reduce model size.

## Research Directions to Explore

The primary research question: **What is the optimal point on the depth-vs-precision frontier?**

Promising directions (roughly in order of expected impact):

1. **Quantization-Aware Training (QAT)**: Add straight-through estimator (STE) fake quantization during warmdown. Try int6, int5, int4 per-weight precision. This lets you fit more layers.

2. **More depth**: With better compression, fit 12, 15, or 20 layers instead of 10. Each additional layer consistently improves language modeling.

3. **Mixed precision per layer**: Critical layers (first, last) at int8; middle layers at int4. Sensitivity varies by layer.

4. **Sliding window eval**: stride=64 gives ~0.03 bpb improvement for free. Already in baseline.

5. **Learning rate tuning**: The optimal LR depends on model size. Wider/deeper models typically need lower LRs.

6. **Longer training sequences**: 2048 or 4096 seq length may help but costs more compute per step.

7. **Architecture tweaks**: MLP expansion ratio, activation function (relu^2 vs swiglu), head dim, GQA ratio.

8. **Custom compression**: Product quantization, learned codebooks, NF4 instead of uniform int quantization.

**Approaches that DON'T work at this scale:**
- Weight sharing / ALBERT-style (tested, fails -- unique params matter more)
- Hypernetworks (too complex to train in 10 min)

## Output Format

The training script prints a summary:
```
---
val_bpb:              1.XXXX
pre_quant_val_bpb:    1.XXXX
artifact_bytes:       XXXXXXX
compressed_model_bytes: XXXXXXX
code_bytes:           XXXXXXX
model_params:         XXXXXXX
num_steps:            XXXX
step_avg_ms:          XXX.XX
training_time_ms:     XXXXXX
peak_memory_mb:       XXXXX
```

Extract key metrics: `grep "^val_bpb:\|^artifact_bytes:\|^num_steps:" run.log`

## Logging Results

Log to `results.tsv` (tab-separated):

```
commit	val_bpb	artifact_mb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved -- use 0.000000 for crashes
3. artifact size in MB, round to .1f -- use 0.0 for crashes
4. status: `keep`, `discard`, `crash`, or `invalid` (over 16MB)
5. short text description of what was tried

Example:
```
commit	val_bpb	artifact_mb	status	description
a1b2c3d	1.2244	15.9	keep	baseline 10L 512dim int8
b2c3d4e	1.2180	14.2	keep	add QAT int6 middle layers
c3d4e5f	1.2300	15.8	discard	increase depth to 12L without QAT (too slow)
d4e5f6g	0.0000	0.0	crash	int4 QAT numerical instability
e5f6g7h	1.2100	17.2	invalid	15L int8 exceeds 16MB limit
```

## The Experiment Loop

LOOP FOREVER:

1. Look at git state and results.tsv to understand current best and what's been tried.
2. Decide what to try next. Prioritize high-impact changes.
3. Modify `train.py` with the experimental change.
4. `git add train.py && git commit -m "<short description>"`
5. Run: `python train.py > run.log 2>&1`
6. Extract results: `grep "^val_bpb:\|^artifact_bytes:\|^num_steps:\|^peak_memory_mb:" run.log`
7. If grep is empty, run crashed. `tail -n 50 run.log` to diagnose. Fix if simple, skip if fundamental.
8. Log to results.tsv.
9. If val_bpb improved AND artifact_bytes <= 16000000: keep (advance branch).
10. If val_bpb worse or artifact invalid: `git reset --hard HEAD~1` (discard).

**Timeout**: Each run should take ~5-6 minutes total. If >10 minutes, kill and treat as failure.

**NEVER STOP**: Once started, do NOT pause to ask the human. Run indefinitely until manually stopped. If stuck, think harder -- re-read the code, try combinations, try radical changes.

**Key principle**: Each experiment should test ONE hypothesis. Don't change 5 things at once -- you won't know what helped.

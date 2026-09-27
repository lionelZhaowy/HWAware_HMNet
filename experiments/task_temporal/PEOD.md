# PEOD: four controlled detection experiments

Implementation is available. Full physical DAT auditing, complete input caching and final full-dataset throughput acceptance are still in progress. Do not interpret bounded smoke results as a completed training run or as deployment acceptance. See each checkout's `artifacts/peod/` for current local evidence.

All four checkouts derive from GEN1 RVT commit `7803484`. The shared implementation is identical; `experiment.toml` selects RGB, DVS RVT, RGB+DVS RVT or RGB+DVS Binary. Each model trains independently. B1 branch specifications, 256-channel projections, Pyramid neck and six-class YOLOX head are the same. Single-stream and dual-stream parameter totals differ.

| Default checkout suffix | modality | representation | parameters |
|---|---|---|---:|
| `RGB_v1.2_T` | rgb | none | 15,403,393 |
| `DVS_v1.2_T_RVT` | dvs | RVT20 | 15,405,841 |
| `RGBDVS_v1.2_T_RVT` | rgbdvs | RVT20 | 20,174,705 |
| `RGBDVS_v1.2_T_Binary` | rgbdvs | Binary2 | 20,172,113 |

Only the DVS branch has Stage3/4 M=2 memory: current and previous label-step summaries, FP32 accumulation/state, BF16 elsewhere, TBPTT=1. RGB is freshly encoded for every sample; RGB-only is stateless but follows exactly the same sequence lanes and flip decisions. No RGB cache exists in the model. Loss, neck and attention mathematics are inherited.

## Data and geometry

The local original dataset is `/data/lab_dataset/RGB_DVS_Fusion/PEOD_orig`. The shared index/input cache is its `preprocessed/hmnet_v12t_240x304_v1` directory. Preparation never replaces original DAT, PNG, CSV or JSON files. Originals are still needed for source verification, raw fallback and audit.

Published rectified RGB is genuinely color, 1280×720, in the published event coordinate system. We use the release's existing alignment; resizing is not a calibration algorithm. Filename-matched COCO images associate with sorted RGB files and the CSV's second column, in integer microseconds. The published timestamp is the label anchor and RGB timestamp (age 0); separate exposure-end metadata was not found. This is release-time causality, not a claim about photon exposure-end time.

All modalities share the exact same manifest, including empty GT and zero events. Official test sequences stay test. From each training condition, sorted sequences numbered tenth/twentieth/etc. within that condition form validation. Current metadata inventory: 88 train sequences / 54,808 frames, 9 val / 4,820, 12 test-challenge / 6,403, 12 test-normal / 5,444. Validation sequences are 010,020,030,043,053,063,073,083,093. The full event scan must finish before these become accepted window indices. Audit found truncated train sequence024 (118 label times past EOF), train sequence025 (all557 label times past EOF), and a partial final record in test-normal sequence019 (after all labels). Repair is in progress using the official sequence_024-5.zip package; no missing event window is accepted as zero input.

RGB 1280×720 is resized by PIL bilinear to 304×171, then padded black with top34/bottom35 to 240×304 and ImageNet normalized. Events map by `x*19//80`, `y*19//80+34` before representation construction. Training boxes are clipped to the source image, zero-area intersections removed, kept as floats, then multiplied by 0.2375 with y offset34. Evaluation keeps original published floating GT and inverse-maps predictions to 1280×720. Predictions outside the source grid are clipped; zero-area predictions are dropped. No GEN1 warmup or small-box filtering is applied.

Exact closed windows are `[t-50000,t]`. A complete physical DAT scan finds enclosing ranges even when timestamps regress; loading filters exact membership and stably sorts. A >2^31 regression fails explicitly instead of guessing wrap epochs. RVT uses the frozen 10-bin, polarity-major implementation (`count_cutoff=10`, `fastmode=True`, including uint8 wrap). Binary is raw-event occupancy, never reconstructed from RVT. Optional compressed caches preserve both arrays exactly. Cache writes have a hard total budget and minimum-free-space guard.

The reset gap is recorded in the accepted manifest, chosen as twice the training-source p99 label interval rounded upward to milliseconds. M=2 refers to label steps, not a fixed 100ms horizon. There is no extra model input padding: convolutional strides use ceiling shapes, yielding /4 60×76, /8 30×38, /16 15×19, /32 8×10.

Sources: [dataset release](https://github.com/bupt-ai-cz/PEOD), [paper](https://arxiv.org/html/2511.08140v1). The paper uses COCO mAP/AP50/AP75. Its published resolution/window/training budgets differ from this experiment's explicitly chosen 240×304/50ms/100 epochs; these runs are an internal controlled experiment, not a reproduction of its leaderboard settings.

## Initialization, training and evaluation

Official B1 checkpoint SHA256: `bf8798aa03ed2bba21fc8d58f43ca9a92096f3f03bd700e1df1dcde11559f499`. Each checkout references the persistent source project's official `pretrained/efficientvit_b1_r224.pth`; no GEN1 trained checkpoint initializes PEOD. Named-component seeds ensure corresponding RGB/DVS non-stem and common neck/head tensors are bitwise equal. Binary adapts canonical20 only after initialization; its stem is official RGB channel mean ×3/2.

Common candidate protocol: seed42, 100 epochs, AdamW lr2e-4 / weight_decay0.01, default betas/epsilon, 5 warmup epochs starting at factor0.1, cosine to2e-6, BF16, accumulation1, single GPU, local BN, deterministic algorithms, TF32 off, no gradient clipping, prefetch1, pin_memory, persistent workers. Validation is every epoch, eval batch32, score0.01, NMS IoU0.65. Main comparison uses epoch100; best uses validation mAP and is a separate table. Updates are recomputed from actual common train samples and batch, never copied from GEN1.

Batch128 has passed a short fusion capacity test (~19.4 GiB allocated /20.3 GiB reserved). Workers2/4/8 sustained pilot benchmarking is ongoing. **Batch/workers are not yet a frozen full-dataset throughput choice.** Four groups must use identical actual microbatch/workers/eval batch and all common hyperparameters. Do not raise single-stream batch separately.

## Commands

Run within the intended independent checkout, for example:

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Det_PEOD_RGBDVS_v1.2_T_RVT
./scripts/hmnet-python scripts/prepare_peod.py
# Optional lossless shared cache; fail closed at total20GiB or free40GiB.
./scripts/hmnet-python scripts/cache_peod.py --limit-sequences 0 --max-cache-gib 20 --reserve-gib 40
```

The other checkouts are in the same parent directory, with suffixes listed above. Configuration defaults select their experiment; all source imports remain checkout-local.

After the complete audit and final common performance configuration are confirmed, the train/resume entry is:

```bash
# Provisional example: replace both batch/workers together for ALL groups if final benchmark changes them.
# Choose a currently idle physical GPU; device within the process stays cuda:0.
CUDA_VISIBLE_DEVICES=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  ./scripts/hmnet-python scripts/task_temporal.py train --epochs 100 \
  --batch 128 --workers 2 --eval-batch 32 --seed 42 --precision bf16
# Resume: repeat exactly the same arguments and add --resume <this-run>/checkpoint.pth.
# For bounded smoke use a separate --output artifacts/peod/smoke_new --stop-after 4.
```

Default official output is `logs/detection/peod_<modality>_<representation>/` (`none` for RGB). Full checkpoint contract includes data/manifest, geometry, modality, representation, initialization, sampling, temporal state and complete schedule. Incompatible resumes are rejected. Checkpoints also save optimizer, RNG, cursor and per-stream memory. Do not resume diagnostic checkpoints for formal training.

```bash
CUDA_VISIBLE_DEVICES=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  ./scripts/hmnet-python scripts/task_temporal.py eval --split test \
  --checkpoint <run>/checkpoint.pth --eval-batch 32 --workers 2 \
  --output artifacts/evaluation/peod_test
./scripts/hmnet-python scripts/export_peod.py --checkpoint <run>/checkpoint.pth \
  --data-root /data/lab_dataset/RGB_DVS_Fusion/PEOD_orig/preprocessed/hmnet_v12t_240x304_v1 \
  --output artifacts/onnx/final
# Add --backbone-only for the four-scale backbone graph.
```

Evaluation writes one complete original-coordinate prediction JSONL used for both COCO scoring and visualization. Test normal/challenge scores come from that same dump. Freeze final checkpoints before comparing. To render four completed dumps without rerunning inference:

```bash
./scripts/hmnet-python scripts/render_peod_comparison.py \
  --data-root /data/lab_dataset/RGB_DVS_Fusion/PEOD_orig/preprocessed/hmnet_v12t_240x304_v1 \
  --split test --dumps <rgb.jsonl> <dvs.jsonl> <fusion_rvt.jsonl> <fusion_binary.jsonl> \
  --output artifacts/comparison --videos
```

Images use 8 uniformly spaced observations. Videos show 10 observations/s and actual acquisition timestamps; they do not claim original-time playback or real-time deployment. Display threshold0.25 affects rendering only; AP uses the stored full postprocessed predictions.

## Validation boundaries

Four groups passed real GPU continuous4 versus split2+resume4 exact comparisons of weights/BN, optimizer, RNG, cursor and DVS state. Boundary checks include empty/single events, both polarities, closed endpoints, timestamp regressions, coordinate collisions, invalid coordinates and RVT overflow. RGB-only never creates event tensors or a temporal state bank. Shared initialization equality is explicitly verified.

Four whole-model and four backbone ONNX graphs are available under each checkout's `artifacts/peod/onnx/`, from four-step smoke weights. All pass ONNX structure checking. **All have strict numerical failures on at least one case; none has deployment acceptance.** Input(s) are `[1,C,240,304]`, output is `[1,1505,11]` before NMS. Event models expose 7 FP32 state tensors: 3×`[1,16,17,16]` and 4×`[1,32,17,16]`. Reports retain independent PyTorch/ORT state trajectories and fixed atol1e-3/rtol1e-4. All-zero means zero normalized input, not raw black RGB.

No full PEOD training or final task accuracy evaluation has been started. GPU short tests and random/early-weight COCO smoke results are not accuracy evidence.

# Asynchronous experiment source references

Pinned during implementation on 2026-09-23. These are algorithm references unless explicitly stated as vendored. No detection labels, box interpolation, NMS, trackers or external training framework were transplanted.

|Project|Pinned revision|Used scope|License|
|---|---|---|---|
|[RVT](https://github.com/uzh-rpg/RVT)|`b80f5683a6e2d5de65d4bde8105d796ccb50dbb1` (reference inspection)|`data/utils/representations.py`; existing frozen local histogram vendor remains unchanged|MIT; existing vendor notice retained|
|[FAOD](https://github.com/Hatins/FAOD-master)|`e8666ca536850807173502e6764423135194cf7f`|`data/ev_img_dataloader/sequence_rnd.py`, `sequence_for_streaming.py`: past-image unpair probability and stream-consistent drift|MIT|
|[UniMatch](https://github.com/LiheYoung/UniMatch)|`0324f3dd4a22128105a5eb135d2586c0bb3cc647`|`unimatch.py`: detached semantic class/confidence, per-pixel ignore mask|MIT|
|[FlexEvent](https://github.com/DylanOrange/flexevent)|`8cf405bb0530a1517ef827c04edfef98f85f4ab0`|README inference/stage descriptions; training code was not released at inspection|MIT|

FAOD drift is adapted to real timestamp lookup and deterministic lane/sequence schedules. Detection label shifts and padding are not used. UniMatch's all-nonignore loss denominator is deliberately replaced by an accepted-pixel mean, separately weighted and logged; its DeepLab model, optimizer, CutMix and framework are not imported. The two frozen stage-1 teachers supply same-time semantic agreement masks. This is an experiment design, not a reproduction claim for either paper.

Local reuse: parent `prepare_dsec_b1.py` alignment/event indexing, `RVTHistogram`, sequence-lane sampler, `TemporalStreams`, `frame_train.py` optimizer/schedule/RNG/strict resume, HMNet segmentation neck/heads, and existing ONNX comparison. BN-safe recomputation is factored from the local CrossModalLiteMLA buffer-swap approach.

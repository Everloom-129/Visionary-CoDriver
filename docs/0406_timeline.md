# Visionary Co-Driver (VCD) Addon Experiments Plan

王杰 

Date 04/06/2025

**Project Timeline:** 2025/03/18 - 2025/04/16

---

## 1. Tasks & Models

### A. Road Scene Perception and Processing

#### 1. **Multiple Object Tracking & Speed Estimation @赵桉**
```bash
VCD/fast_vision
```

- **Models**: YOLOX + ByteTrack Backbone
- **Dataset**: JAAD, BDD100K (MOT subset) [video at 30Hz]
- **Scale**: \~3k annotated sequences
- **Status**: ✅ Completed Accuracy & Runtime on JAAD subset (1.5k sequences)
    
| model           | mAP50 | TP(%) | FP(%) | FN(%) | TN(%) | Prec(%) | Rec(%) | FPS on RTX3090 |
| -------------- | ----| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| YOLOX-x        | 55.6 | 51.9 | 9.8 | 38.3 | 0.0 | 84.1 | 57.5 | 56.82 |


| model              | MOTA | IDF1 | HOTA | FPS on RTX3090 |
| ----------------- | ---- | ---- | ---- | -------------- |
| bytetrack_x_mot17 | 74.7 | 80.3 | 73.1 | 32.17          |



------
- **TODO**: 
        - [ ] 10 video sequences on BDD100K (used in the paper)
        - [ ] speed estimation script (TO BE provided by @王杰)
- **ETA**: 04/09/2025




#### 2. **Road vs Sidewalk Segmentation @王杰**
```bash
VCD/slow_vision
```
- **Models**: Grounding DINO + SAM  = DINO-X
- **Dataset**: JAAD [videos at 2Hz]
- **Scale**: \~5k images with dense annotations
- **Status**: 🔄 Refactor the code for open source
    - TODO:
        - [ ] 10 video sequences on BDD100K (used in the paper)
        - [ ] slow-fast fusion 
- **ETA**: 04/08/2025

#### 3. **Distance and Behavior Estimation**
```bash
VCD/slow_vision
```
- **Models**: DPT + Mask Computation -> DepthPro for depth estimation
- **Dataset**: JAAD (for pedestrian)
- **Scale**: \~10k frames
- **Status**: ⏳ DPT depth map calibrated, behavior classifier TBD
- **ETA**: 04/11/2025

---

## 2. Integral Scene Description (Fusion Pipeline)
```bash
VCD/scene_fusion.py
```
- **Inputs**: Outputs from tracking, segmentation, depth modules
- **Output**: Unified coordinate-space JSON (for GPT)
- **Testing Plan**: Validate internal consistency and edge cases
- **Status**: 🔄 Upgrading
- **ETA**: 04/12/2025

---

## 3. Risk Analysis Module
```bash
VCD/agent/
```
- **Model**: GPT-3.5-16k-0613
- **Prompting**: System Prompt + Few-shot Examples (curated driving risks)
- **Input**: Integral Scene Description
- **Output**: risk ID + pedestrian + text description
- **Datasets**: Expert 
- **Status**: ✅ Get code working but haven't committed to the repo
- **ETA**: 04/13/2025

---

## 4. System Integration Tests

### A. Full Pipeline (Module Integration)

- **Tasks**: Sequence-through inference + Unity interface visual
- **Datasets**: 20 annotated videos from JAAD + custom screen recordings
- **Status**: ⏳ In progress
- **ETA**: 04/14/2025

### B. HUD Display Consistency

- **Validation**: Unity screen map match with risk text overlay
- **Status**: 🔜 Planned
- **ETA**: 04/15/2025

---

## 5. Final Deliverables

- **Report Update**: Documentation of modules, model choices, ablations
- **Demo**: 10 sample scenes with pipeline walkthrough (video + text)
- **Due**: 04/16/2025


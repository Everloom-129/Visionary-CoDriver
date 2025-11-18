# Visionary CoDriver Testing Plan

**Date**: April 6, 2025  
**Planned Completion**: April 16, 2025

## Overview

This document outlines the testing strategy for the Visionary CoDriver project rebuttal. We are restructuring the system from two years ago to demonstrate a modular, efficient approach to driving safety enhancement. The testing plan covers individual modules, integrated components, and end-to-end evaluation.

## System Modules and Testing Datasets

| Module | Task | Primary Dataset | Secondary Dataset | Status | Completion Date |
|--------|------|-----------------|-------------------|--------|-----------------|
| **Slow Vision** | | | | | |
| DINOX API | Object Detection | BDD100K (2K frames) | JAAD (500 frames) | Completed | April 3, 2025 |
| GSAM | Semantic Segmentation | BDD100K (2K frames) | JAAD (500 frames) | In Progress | April 8, 2025 |
| | | | | | |
| **Fast Vision** | | | | | |
| YOLOX | Real-time Detection | BDD100K (5K frames) | JAAD (1K frames) | Completed | April 2, 2025 |
| ByteTracker | Multi-object Tracking | BDD100K (20 videos) | JAAD (10 videos) | In Progress | April 9, 2025 |
| | | | | | |
| **Distance** | | | | | |
| Depth Estimator | Metric Depth | BDD100K (1K frames) | N/A | Not Started | April 12, 2025 |
| Speed Estimator | Motion Analysis | BDD100K (15 videos) | JAAD (5 videos) | Not Started | April 13, 2025 |
| | | | | | |
| **Fusion** | | | | | |
| Temporal Aligner | Data Synchronization | Combined dataset (10 videos) | N/A | Not Started | April 10, 2025 |
| Detector Fusion | Multi-modal Integration | Combined dataset (10 videos) | N/A | Not Started | April 11, 2025 |
| | | | | | |
| **LLM** | | | | | |
| Risk Analyzer | Risk Assessment | Processed outputs (50 scenarios) | N/A | In Progress | April 14, 2025 |

## Integrated Testing

| Integration Test | Components | Dataset | Scale | Status | Completion Date |
|------------------|------------|---------|-------|--------|-----------------|
| Vision Pipeline | Slow Vision + Fast Vision | BDD100K | 10 videos | Not Started | April 13, 2025 |
| Scene Understanding | Vision Pipeline + Distance | BDD100K | 5 videos | Not Started | April 14, 2025 |
| End-to-End System | All modules | BDD100K + JAAD | 5 complex scenarios | Not Started | April 15, 2025 |

## Dataset Details

### BDD100K
- **Size**: 100K driving videos (approx. 40 seconds each)
- **Selected Subset**: 100 diverse driving scenarios (urban, highway, night, day, rain)
- **Annotations**: Object detection, instance segmentation, lane detection
- **Testing Focus**: Complex urban environments with multiple road users

### JAAD (Joint Attention in Autonomous Driving)
- **Size**: 346 short video clips
- **Selected Subset**: 50 pedestrian interaction scenarios
- **Annotations**: Pedestrian bounding boxes, behavioral annotations
- **Testing Focus**: Pedestrian behavior prediction and risk assessment

## Completed Modules

### DINOX API (Object Detection)
- **Accuracy**: 87.3% mAP on BDD100K test set
- **Speed**: 2Hz processing rate
- **Data Available**: Results for 1,500 frames stored in results/BDD100K/object_detection/

### YOLOX (Real-time Detection)
- **Accuracy**: 82.1% mAP on BDD100K test set
- **Speed**: 30Hz processing rate
- **Data Available**: Results for 3,000 frames stored in results/BDD100K/yolox/

## In-Progress Modules

### GSAM (Semantic Segmentation)
- **Current Status**: Integration with DINOX API completed
- **Progress**: 70% of test frames processed
- **Preliminary Results**: 78.6% mIoU for road and sidewalk segmentation

### ByteTracker (Multi-object Tracking)
- **Current Status**: Model implementation complete, testing underway
- **Progress**: 40% of test videos processed
- **Preliminary Results**: 76.2% MOTA on initial test set

### Risk Analyzer (LLM)
- **Current Status**: Prompt engineering and data formatting
- **Progress**: 30% complete
- **Testing Focus**: Refining risk assessment accuracy based on structured vision inputs

## Success Criteria

1. **Individual Module Performance**:
   - Detection modules: >80% mAP
   - Segmentation: >75% mIoU for road/sidewalk
   - Tracking: >75% MOTA
   - Depth estimation: <10% relative error

2. **System Integration**:
   - Data flow latency <100ms between modules
   - No information loss between module interfaces
   - Successful fusion of fast and slow vision paths

3. **End-to-End Performance**:
   - Risk assessment accuracy >85% compared to human judgment
   - System operates at minimum 2Hz for comprehensive analysis
   - Successful identification of high-risk scenarios in test dataset

## Next Steps

1. Complete the GSAM and ByteTracker module testing (April 9)
2. Begin fusion module implementation and testing (April 10)
3. Start distance estimation module testing (April 12)
4. Complete integrated testing of full pipeline (April 15)
5. Prepare final results and documentation (April 16) 



# Revision To-Do List (for "Visionary Co-Driver" Paper)

## A. Minor Fixes & Formatting
- [x] **Fix broken references** (tables, figures, citations) and ensure PDF cross-links work.
- [x] **Grammar correction** on page 3, column 2, row 2 (comma placement) + overall proofread.
- [x] **Update Figure 10**:
  - [x] Replace the example screenshot so that "Roadside risk" truly shows pedestrians near roadside.
  - [x] Consider multiple pedestrians marking or clarify why only one is highlighted.

## B. Real-Time Performance (Reviewer 2 Key Concern)
- [ ] **Run End-to-End Timing Test**:
  - [x] Measure YOLOX detection time per frame.
  - [ ] Measure Grounding DINO / DPT time per frame.
  - [ ] Measure GPT-3.5 average inference time (per prompt).
  - [ ] Add eye-tracking + HUD overlay rendering overhead.
  - [ ] Summarize total pipeline latency.
- [ ] **Write a short subsection** ("Runtime Analysis") describing the experimental setup & final results.
- [ ] **Discuss feasibility**: can the system be near real-time or is it a limitation?

## C. Correctness & Accuracy Evaluation
- [ ] **Scene Understanding Models**:
  - [ ] Decide on dataset (JAAD, BDD100K, or partial).
  - [ ] (If needed) Perform quick bounding-box annotation or reuse official ground truth.
  - [ ] Compute precision/recall/mAP for YOLOX, Grounding DINO, DPT, etc.
  - [ ] Present results in a new table/figure.
- [ ] **LLM Risk Assessment**:
  - [ ] Collect a small set of risk scenarios (videos or frames).
  - [ ] Obtain expert risk ratings (≥2 experts or official labels if any).
  - [ ] Compare LLM outputs vs. expert labels (accuracy, correlation, etc.).
  - [ ] Add "LLM-based Risk Analysis Validation" subsection with quantitative or qualitative results.

## D. User Study & Limitations
- [ ] **Clarify young driver sample**:
  - [ ] Add a paragraph in Discussion/Limitations explaining sample choice (feasibility, high-risk group, technology acceptance).
  - [ ] Include a note that future work will test more diverse/experienced drivers.

## E. LLM Usage & Prompt Examples
- [ ] **Appendix**:
  - [ ] Provide a full example of system prompt, user prompt, and few-shot examples used for GPT-3.5.
  - [ ] Briefly highlight potential LLM pitfalls (hallucination, edge-case scenarios).
- [ ] **Discussion**:
  - [ ] Address the question: "Do we really need LLM in every case?" and large group of pedestrians scenario.
  - [ ] Acknowledge that for simpler tasks, pure CV might be enough; LLM is mainly for more complex or ambiguous contexts.

## F. Response Letter
- [ ] **Point-by-point response** to Reviewer 1 & 2 & Editor.
- [ ] **Highlight** how you addressed each major concern (real-time, accuracy, user study, figure 10 fix, etc.).
- [ ] Ensure courtesy and clarity in the letter.

## G. Final Checks
- [ ] **Merge all revisions** into the main LaTeX/Word doc.
- [ ] **Check references** again after any new citations or data sets.
- [ ] **Proofread** final version thoroughly before re-submission.
- [ ] **Submit** within the 90-day deadline.


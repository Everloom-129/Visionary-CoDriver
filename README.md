# Visionary CoDriver
Tony Wang
2025-03-18

## Overview

This project is a comprehensive system designed to enhance driving safety and efficiency. It integrates various components to provide a complete solution for autonomous driving.

## Features

1. Semantic Segmentation of Road and Sidewalk
2. Open world Object Detection and Multi-object Tracking for pedestrians and vehicles
3. Metric Depth Estimation
4. Risk Assessment via LLM

## Installation

```bash
pip install -r requirements.txt
```

### Download weights

```bash
cd config
bash download_ckpt.sh
```

### Test
1. Test GSAM Detector
```bash
cd VCD/slow_vision
python gsam_detector.py
```
2. Test Roadside Analyzer
```bash
cd VCD/slow_vision
python roadside_analyzer.py
```


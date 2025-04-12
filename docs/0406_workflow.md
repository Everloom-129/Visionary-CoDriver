# 重构Visionary CoDriver代码

创建一个清晰、模块化的系统架

## 需要做的工作

1. **精简文件结构**：创建一个更有组织的源代码目录
2. **改进GSAM模块**：使用DINOX API重构视觉处理模块
3. **明确数据流**：从输入数据集到LLM分析的完整流程
4. **文档化**：创建清晰的文档说明系统架构和使用方法

## File Structure

```
Visionary-CoDriver/ 
├── VCD/
│   ├── slow_vision/ # 2Hz
│   │   ├── api_client.py            #  DINOX API
│   │   ├── DINOX_detector.py        # 物体检测模块(使用DINOX API)
│   │   └── roadside_analyzer.py     # GSAM测试
│   │   └── __init__.py      
│   ├── fast_vision/ # 30Hz
│   │   ├── ByteTrack/           # YOLOX + ByteTrack
│   │   ├── bytetracker.py       # MOT
│   │   ├── speed_estimator.py         # 速度估计模块
│   │   └── depth_estimator.py         # 深度估计模块
│   │   └── __init__.py      
│   ├── fusion/
│   │   ├── temporal_aligner.py  # 时间对齐
│   │   └── detector_fusion.py   # 检测结果融合
│   │   └── __init__.py      
    
│   ├── LLM/       # 2Hz
│   │   ├── data_parser.py   # 原始数据处理
│   │   ├── prompt.md        # 提示词
│   │   └── risk_analyzer.py # 风险评估
│   │   └── __init__.py      
│   └── utils/
│   │   ├── SAM_utility.py   # SAM工具
│   │   ├── visualization.py # 可视化工具
│   │   ├── time_utils.py    # decorator for time measurement
│   │   └── __init__.py      
│   └── __init__.py      
├── config/
│   └── default.yaml         # 配置文件
│   └── vision.yaml          # 视觉处理配置
├── data/
│   ├── JAAD/                # 数据集
│   └── BDD100K/             # 数据集
├── docs/
│   ├── architecture.md      # 系统架构文档
│   └── workflow.md          # 工作流程文档
├── results/
│   ├── JAAD/                
│   └── BDD100K/             
├── main.py                  # 主程序入口, 可选
└── README.md                # 项目说明
```

## 工作流程

1. **数据输入**：从JAAD或BDD100K加载图像/视频数据
2. **视觉处理**：
   - 使用DINOX API进行物体检测
   - 使用SAM进行精确分割
   - 使用DepthPro进行深度估计
3. **多目标跟踪**：跟踪视频中的行人和车辆
4. **数据集成**：将检测、分割、深度和跟踪信息整合
5. **结构化输出**：生成中间结构化数据
6. **LLM分析**：将结构化数据输入NLP.py进行风险分析


## 工作流文档 (workflow.md)

需要在docs/workflow.md中详细说明完整的数据处理流程：

1. **数据加载**：从数据集加载视频帧
2. **视觉处理阶段**：
   - 物体检测：识别道路、人行道、行人和车辆
   - 语义分割：生成精确的像素级掩码
   - 深度估计：计算场景深度
3. **追踪阶段**：
   - 行人追踪：跨帧跟踪行人位置
   - 速度/行为估计：分析行人移动模式
4. **结构化输出阶段**：
   - 生成JSON格式的结构化数据
   - 包含行人位置、关系和风险指标
5. **LLM分析阶段**：
   - 将结构化数据输入至NLP模块
   - 生成风险评估和行为分析


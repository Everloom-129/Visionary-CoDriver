import cv2
import numpy as np
import os
import sys
import argparse
from collections import defaultdict


def visualize_tracking_results(video_path, person_tracking_file, car_tracking_file, output_path=None):
    """
    可视化行人和车辆目标追踪结果，在原始视频上绘制边界框、track_id和速度状态。
    
    参数:
        video_path (str): 原始视频文件路径
        person_tracking_file (str): 包含行人追踪结果的txt文件路径
        car_tracking_file (str): 包含车辆追踪结果的txt文件路径
        output_path (str): 输出视频的路径，默认为在原视频所在目录添加"_visualized"后缀
    """
    # 如果未指定输出路径，创建默认输出路径
    if output_path is None:
        filename, ext = os.path.splitext(video_path)
        output_path = f"{filename}_visualized{ext}"
    
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height}, {fps} FPS, {total_frames} 帧")
    
    # 创建输出视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 也可以使用 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 读取行人跟踪结果文件
    person_tracks_by_frame = defaultdict(list)
    
    try:
        with open(person_tracking_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 11:  # 确保有足够的字段，包括速度标记
                    frame_id = int(float(parts[0]))
                    track_id = int(float(parts[1]))
                    top = float(parts[2])
                    left = float(parts[3])
                    width_box = float(parts[4])
                    height_box = float(parts[5])
                    speed_flag = int(parts[10])  # 最后一个字段是速度标记
                    
                    person_tracks_by_frame[frame_id].append({
                        'track_id': track_id,
                        'bbox': [left, top, width_box, height_box],
                        'speed': 'slow' if speed_flag == 0 else 'fast',
                        'type': 'person'
                    })
    except Exception as e:
        print(f"读取行人跟踪文件时出错: {e}")
        cap.release()
        out.release()
        return
    
    print(f"加载了 {len(person_tracks_by_frame)} 帧的行人跟踪数据")
    
    # 读取车辆跟踪结果文件
    car_tracks_by_frame = defaultdict(list)
    
    try:
        with open(car_tracking_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 11:  # 确保有足够的字段，包括速度标记
                    frame_id = int(float(parts[0]))
                    track_id = int(float(parts[1]))
                    top = float(parts[2])
                    left = float(parts[3])
                    width_box = float(parts[4])
                    height_box = float(parts[5])
                    speed_flag = int(parts[10])  # 最后一个字段是速度标记
                    
                    car_tracks_by_frame[frame_id].append({
                        'track_id': track_id,
                        'bbox': [left, top, width_box, height_box],
                        'speed': 'slow' if speed_flag == 0 else 'fast',
                        'type': 'car'
                    })
    except Exception as e:
        print(f"读取车辆跟踪文件时出错: {e}")
        cap.release()
        out.release()
        return
    
    print(f"加载了 {len(car_tracks_by_frame)} 帧的车辆跟踪数据")
    
    # 合并行人和车辆的帧数据，用于处理
    all_frames = set(list(person_tracks_by_frame.keys()) + list(car_tracks_by_frame.keys()))
    
    # 为不同track_id分配不同颜色
    np.random.seed(42)  # 保证每次运行颜色一致
    person_color_map = {}  # 行人的颜色映射
    car_color_map = {}     # 车辆的颜色映射
    
    # 默认颜色 - 使用整数值
    person_base_color = (0, 255, 0)  # 绿色为行人基础颜色
    car_base_color = (0, 0, 255)     # 红色为车辆基础颜色
    
    # 处理每一帧
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 在当前帧上绘制所有行人目标
        if frame_idx in person_tracks_by_frame:
            for track in person_tracks_by_frame[frame_idx]:
                track_id = track['track_id']
                
                # 为track_id分配固定的颜色，在行人的基础颜色上变化
                if track_id not in person_color_map:
                    person_color_map[track_id] = (
                        int(np.clip(person_base_color[0] + np.random.randint(-50, 50), 0, 255)),
                        int(np.clip(person_base_color[1] + np.random.randint(-50, 50), 0, 255)),
                        int(np.clip(person_base_color[2] + np.random.randint(-50, 50), 0, 255))
                    )
                
                color = person_color_map[track_id]
                bbox = track['bbox']
                speed = track['speed']
                obj_type = track['type']
                
                # 绘制边界框
                y, x, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # 准备标签文本
                label = f"{obj_type} ID:{track_id} ({speed})"
                
                # 选择速度标签的颜色
                speed_color = (0, 255, 0) if speed == 'slow' else (0, 0, 255)  # 绿色表示慢，红色表示快
                
                # 绘制背景矩形以增强文本可见性
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
                
                # 绘制对象类型、ID和速度标签
                cv2.putText(
                    frame, 
                    f"{obj_type} ID:{track_id}", 
                    (x, y - baseline - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1
                )
                
                # 在边界框下方显示速度
                cv2.putText(
                    frame, 
                    speed, 
                    (x, y + h + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    speed_color, 
                    2
                )
        
        # 在当前帧上绘制所有车辆目标
        if frame_idx in car_tracks_by_frame:
            for track in car_tracks_by_frame[frame_idx]:
                track_id = track['track_id']
                
                # 为track_id分配固定的颜色，在车辆的基础颜色上变化
                if track_id not in car_color_map:
                    car_color_map[track_id] = (
                        int(np.clip(car_base_color[0] + np.random.randint(-50, 50), 0, 255)),
                        int(np.clip(car_base_color[1] + np.random.randint(-50, 50), 0, 255)),
                        int(np.clip(car_base_color[2] + np.random.randint(-50, 50), 0, 255))
                    )
                
                color = car_color_map[track_id]
                bbox = track['bbox']
                speed = track['speed']
                obj_type = track['type']
                
                # 绘制边界框
                y, x, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # 准备标签文本
                label = f"{obj_type} ID:{track_id} ({speed})"
                
                # 选择速度标签的颜色
                speed_color = (0, 255, 0) if speed == 'slow' else (0, 0, 255)  # 绿色表示慢，红色表示快
                
                # 绘制背景矩形以增强文本可见性
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
                
                # 绘制对象类型、ID和速度标签
                cv2.putText(
                    frame, 
                    f"{obj_type} ID:{track_id}", 
                    (x, y - baseline - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1
                )
                
                # 在边界框下方显示速度
                cv2.putText(
                    frame, 
                    speed, 
                    (x, y + h + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    speed_color, 
                    2
                )
        
        # 在左上角显示帧索引
        cv2.putText(
            frame, 
            f"Frame: {frame_idx}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # 保存处理后的帧
        out.write(frame)
        
        # 显示进度
        if frame_idx % 100 == 0:
            print(f"处理帧: {frame_idx}/{total_frames}")
            
        frame_idx += 1
    
    # 释放资源
    cap.release()
    out.release()
    
    print(f"可视化完成。输出视频保存到: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='可视化行人和车辆目标追踪结果')
    parser.add_argument('video_path', help='原始视频文件路径')
    parser.add_argument('person_tracking_file', help='包含行人追踪结果的txt文件路径')
    parser.add_argument('car_tracking_file', help='包含车辆追踪结果的txt文件路径')
    parser.add_argument('--output', '-o', help='输出视频路径')
    
    args = parser.parse_args()
    
    visualize_tracking_results(args.video_path, args.person_tracking_file, args.car_tracking_file, args.output)
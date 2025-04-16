import numpy as np
import os
import sys

def process_tracking_file(file_path, threshold=0.1):
    """
    处理跟踪文件并添加速度分类（0表示慢，1表示快）。
    
    参数:
        file_path (str): 跟踪文件的路径。
        threshold (float): 慢/快分类的阈值。
                          相对移动大于此阈值将被归类为快速。
    """
    try:
        # 读取文件
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        print(f"读取了 {len(lines)} 行数据")
        
        # 解析数据
        data = []
        for line in lines:
            items = line.strip().split(',')
            if len(items) >= 10:  # 确保有足够的元素
                try:
                    frame_id = int(float(items[0]))
                    track_id = int(float(items[1]))
                    top = float(items[2])
                    left = float(items[3])
                    width = float(items[4])
                    height = float(items[5])
                    
                    data.append({
                        'line': line.strip(),
                        'frame_id': frame_id,
                        'track_id': track_id,
                        'top': top,
                        'left': left,
                        'width': width,
                        'height': height,
                        'center_x': left + width/2,
                        'center_y': top + height/2
                    })
                except (ValueError, IndexError) as e:
                    print(f"解析行时出错: {line.strip()}. 错误: {e}")
        
        print(f"成功解析了 {len(data)} 条跟踪记录")
        
        # 按track_id分组
        tracks = {}
        for item in data:
            if item['track_id'] not in tracks:
                tracks[item['track_id']] = []
            tracks[item['track_id']].append(item)
        
        # 按frame_id对每个轨迹排序
        for track_id in tracks:
            tracks[track_id].sort(key=lambda x: x['frame_id'])
        
        print(f"数据分组为 {len(tracks)} 个唯一轨迹")
        
        # 计算每条记录的速度
        speed_classification = {}
        for track_id, track_data in tracks.items():
            for i in range(len(track_data)):
                if i == 0:  # 该轨迹的第一帧
                    # 没有前一帧可比较，所以归类为慢速
                    speed_classification[(track_data[i]['frame_id'], track_id)] = 0
                else:
                    current = track_data[i]
                    previous = track_data[i-1]
                    
                    # 计算移动距离
                    dx = current['center_x'] - previous['center_x']
                    dy = current['center_y'] - previous['center_y']
                    
                    # 计算相对移动（按物体大小归一化）
                    # 使用宽度和高度的平均值进行归一化
                    object_size = (current['width'] + current['height']) / 2
                    if object_size > 0:
                        relative_movement = np.sqrt(dx**2 + dy**2) / object_size
                    else:
                        relative_movement = 0
                    
                    # 归类为慢速或快速
                    if relative_movement > threshold:
                        speed_classification[(current['frame_id'], track_id)] = 1  # 快速
                    else:
                        speed_classification[(current['frame_id'], track_id)] = 0  # 慢速
        
        # 创建带有速度分类的新行
        new_lines = []
        for line in lines:
            items = line.strip().split(',')
            if len(items) >= 10:
                try:
                    frame_id = int(float(items[0]))
                    track_id = int(float(items[1]))
                    
                    # 获取速度分类
                    speed = speed_classification.get((frame_id, track_id), 0)
                    
                    # 创建带有速度分类的新行
                    new_line = line.strip() + ',' + str(speed) + '\n'
                    new_lines.append(new_line)
                except (ValueError, IndexError):
                    # 如果无法解析，则保持行不变
                    new_lines.append(line)
            else:
                # 如果不符合预期格式，则保持行不变
                new_lines.append(line)
        
        # 写入原始文件
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        
        print(f"处理完成。速度分类已添加到 {file_path}")
        
    except Exception as e:
        print(f"处理文件时出错: {e}")
        sys.exit(1)

def get_all_files_in_directory(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python script.py <tracking_file_dir> [threshold]")
        sys.exit(1)
    
    file_dir = sys.argv[1]

    from tqdm import tqdm
    for file_path in tqdm(get_all_files_in_directory(file_dir)):
        
        if len(sys.argv) >= 3:
            threshold = float(sys.argv[2])
            process_tracking_file(file_path, threshold)
        else:
            process_tracking_file(file_path)  # 使用默认阈值0.1

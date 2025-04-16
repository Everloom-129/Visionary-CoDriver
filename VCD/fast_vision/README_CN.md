## 使用方法

### 使用 YOLOX+ByteTrack 推理 MOT 结果

```bash
cd ByteTrack
pip install -e .
cd ..

python3 bytetracker.py video --path <your_path_to_JAAD_dataset>/JAAD_clips -f ByteTrack/exps/default/yolox_x.py -c <your_path_to_yolox-x_pre-trained_weight> --save_result
```

结果的 txt 文件会保存至 `YOLOX_outputs` 目录下。

### 用 relative pixel 计算出每个 object 的运动速度

```bash
python3 track_speed_classifier.py <your_path_to_MOT_results_directory> [threshold]
```

该指令会读取该目录下所有的 txt 文件，并根据 threshold 来判定是 fast(用1标识) 还是 slow(用0标识)，结果会原地保存在 txt 文件的每一行的最后。

### 可视化结果

```bash
python visualize_tracking.py <your_path_to_video> <your_path_to_person_MOT_txt_file> <your_path_to_car_MOT_txt_file> -o <output_path>
```
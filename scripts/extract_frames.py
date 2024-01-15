import os
import sys
import subprocess
import concurrent.futures
from tqdm import tqdm

MAX_THREADS = 8  # 最大线程数
FRAMES_PER_VIDEO = 10  # 每个视频抽取的帧数

def extract_frames(video_path, output_dir):
    video_name = os.path.basename(video_path)
    output_pattern = os.path.join(output_dir, f"{video_name}_frame_%d.jpg")
    extract_command = f'ffmpeg -hide_banner -loglevel error -i "{video_path}" -vf "select=not(mod(n\,{int(FRAMES_PER_VIDEO)}))" -vsync vfr -q:v 2 "{output_pattern}"'
    subprocess.run(extract_command, shell=True)

def process_folder(input_folder, output_folder):
    video_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):  # 可根据需要调整支持的视频格式
                video_files.append(os.path.join(root, file))

    with tqdm(total=len(video_files), unit='video', ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            for video_file in video_files:
                future = executor.submit(extract_frames, video_file, output_folder)
                future.add_done_callback(lambda f: pbar.update(1))

            # 等待所有任务完成
            for _ in concurrent.futures.as_completed(list(concurrent.futures.as_completed(future_list))):
                pass

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("请提供输入文件夹路径和输出文件夹路径")
        sys.exit(1)

    input_folder = sys.argv[1]  # 输入文件夹路径
    output_folder = sys.argv[2]  # 输出文件夹路径

    process_folder(input_folder, output_folder)
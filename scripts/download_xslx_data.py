import os
import sys
import concurrent.futures
import requests
import pandas as pd
from tqdm import tqdm

MAX_THREADS = 100  # 最大线程数

def download_file(url, filename):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as file:
                file.write(response.content)
            return filename
        else:
            return None
    except requests.exceptions.RequestException as e:
        return None

def check_file_exists(filename):
    return os.path.exists(filename)

def process_table_data(table_data, output_dir):
    downloaded_files = []  # 已下载的文件列表
    total_files = 0  # 总文件数量

    # 计算总文件数量
    for _, row in table_data.iterrows():
        video_url = row['videos_addr']
        photo_urls = row[['evidence1', 'evidence2', 'evidence3']]
        total_files += 1 + sum(pd.notnull(photo_url) for photo_url in photo_urls)

    with tqdm(total=total_files, unit='file') as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            for _, row in table_data.iterrows():
                video_url = row['videos_addr']
                photo_urls = row[['evidence1', 'evidence2', 'evidence3']]

                # 下载视频文件
                video_filename = os.path.join(output_dir, video_url.split('/')[-1])
                if not check_file_exists(video_filename):
                    future = executor.submit(download_file, video_url, video_filename)
                    future.add_done_callback(lambda f: pbar.update(1))  # 更新进度条
                    downloaded_files.append(future)
                else:
                    pbar.update(1)
                    print(f"文件已存在，跳过下载：{video_filename}")

                # 下载照片文件
                for i, photo_url in enumerate(photo_urls):
                    if pd.notnull(photo_url):
                        photo_filename = os.path.join(output_dir, f"photo_{row['key']}_{i}.jpg")
                        if not check_file_exists(photo_filename):
                            future = executor.submit(download_file, photo_url, photo_filename)
                            future.add_done_callback(lambda f: pbar.update(1))  # 更新进度条
                            downloaded_files.append(future)
                        else:
                            pbar.update(1)
                            print(f"文件已存在，跳过下载：{photo_filename}")

            # 等待所有下载任务完成
            for future in concurrent.futures.as_completed(downloaded_files):
                filename = future.result()
                if filename is not None:
                    print(f"下载文件：{filename}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("请提供输入表格路径和下载输出文件夹路径")
        sys.exit(1)

    file_path = sys.argv[1]  # 输入表格路径
    output_dir = sys.argv[2]  # 下载输出文件夹路径
    
    # 读取表格数据
    df = pd.read_excel(file_path)

    # 调用 process_table_data 函数进行处理
    process_table_data(df, output_dir)
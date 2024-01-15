import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
from data.constant import culane_row_anchor, tusimple_row_anchor
import argparse
from data.dataloader import get_dataloader
import sys
from utils.dist_utils import dist_print, dist_tqdm


status_list = ['normal','abnormal']
lighting_list = ['day', 'night', 'others']
weather_list = ['sunny', 'rain', 'snow', 'cloudy', 'unknown']
environmet_list = ['internal', 'closed', 'urban', 'tunnle', 'others']

if __name__ == "__main__":

    plot_loader = get_dataloader(1, "", 200, sys.argv[1], True, False, False, 12, 800, 480)

    for index, data in enumerate(dist_tqdm(plot_loader)):
        
        if(index > 5000):
            break

        # 获取标签
        img_ori, img, status_label, lighting_label, weather_label, road_construction_label, cls_label, type_label, func_label, color_label = data

        img_ori = img_ori.numpy()[0]

        # 遍历每条车道线
        for i in range(12):
            # 遍历纵向每一个 grid
            for k in range(20):
                # 当这个点检测到有车道线时
                if status_label == 0:
                    # 取出纵向这个点的对应的 color、func、type 输出结果
                    max_color_index = color_label[0, 19 - k, i]
                    max_func_index = func_label[0, 19 - k, i]
                    max_type_index = type_label[0, 19 - k, i]
                    max_xValue_index = cls_label[0, 19 - k, i]

                    col_sample = np.linspace(0, 800 - 1, 200)
                    col_sample_w = col_sample[1] - col_sample[0]

                    # 获取该 grid 的中心点画圆
                    ppp = (int(max_xValue_index * col_sample_w * img_ori.shape[1] / 800) - 1, int(img_ori.shape[0] * (culane_row_anchor[20-1-k] / 480)) - 1 )

                    if max_color_index == 1:
                        color = (0, 255, 255)
                    elif max_color_index == 2:
                        color = (255, 255, 255)
                    elif max_color_index == 3:
                        color = (0, 0, 255)
                    elif max_color_index == 4:
                        color = (255, 0, 0)
                    elif max_color_index == 5:
                        color = (144, 144, 144)
                    else:
                        color = (0, 0, 0);

                    # 虚线画空心点
                    if max_type_index == 1:
                        cv2.circle(img_ori, ppp, 10, color, 10)
                    # 实线画实心点
                    elif max_type_index == 2:
                        cv2.circle(img_ori, ppp, 15, color, -1)
                    # 网格线画 x
                    elif max_type_index == 3:
                        cv2.line(img_ori, (ppp[0] - 20, ppp[1] - 20), (ppp[0] + 20, ppp[1] + 20), color, thickness=5)
                        cv2.line(img_ori, (ppp[0] - 20, ppp[1] + 20), (ppp[0] + 20, ppp[1] - 20), color, thickness=5)
                    # 导流线画 ^
                    elif max_type_index == 4:
                        cv2.line(img_ori, (ppp[0], ppp[1]), (ppp[0] - 20, ppp[1] + 20), color, thickness=5)
                        cv2.line(img_ori, (ppp[0], ppp[1]), (ppp[0] + 20, ppp[1] + 20), color, thickness=5)
                    # 特殊虚线画 ..
                    elif max_type_index == 5:
                        cv2.circle(img_ori, ppp, 5, color, -1)
                        cv2.circle(img_ori, (ppp[0] + 10, ppp[1]), 5, color, -1)
                    # 特殊实线画 =
                    elif max_type_index == 6:
                        cv2.line(img_ori, (ppp[0] - 10, ppp[1]), (ppp[0] + 10, ppp[1]), color, thickness=5)
                        cv2.line(img_ori, (ppp[0] - 10, ppp[1] + 10), (ppp[0] + 10, ppp[1] + 10), color, thickness=5)
                    # 道路边缘画 |
                    elif max_type_index == 7:
                        cv2.line(img_ori, (ppp[0], ppp[1] - 10), (ppp[0], ppp[1] + 10), color, thickness=10)
                    # 脑部线画超细空心圆
                    elif max_type_index == 8:
                        cv2.circle(img_ori, ppp, 15, color, 2)

                    if max_func_index == 2:
                        cv2.putText(img_ori, 'DX', ppp, cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

                    if max_func_index == 5:
                        cv2.putText(img_ori, 'YJ', ppp, cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

                    if max_func_index == 6:
                        cv2.putText(img_ori, 'BS', ppp, cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

                    if max_func_index == 7:
                        cv2.putText(img_ori, 'BL', ppp, cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
                    

        cv2.putText(img_ori, status_list[status_label], (100, 200), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5)
        cv2.putText(img_ori, lighting_list[lighting_label], (100, 300), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5)
        cv2.putText(img_ori, weather_list[weather_label], (100, 400), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5)
        cv2.putText(img_ori, environmet_list[road_construction_label], (100, 500), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5)
        cv2.imwrite("./out/" + str(index) + ".png", img_ori)
        
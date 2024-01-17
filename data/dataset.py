import torch
from PIL import Image
import os
import pdb
import numpy as np
import cv2
from data.mytransforms import find_start_pos
import random
import math


class LaneDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, augment=False, num_x_grid=50, row_anchor=None, num_lanes=12, resized_width=800, resized_height=480, use_aux=False):
        super(LaneDataset, self).__init__()
        self.augment = augment
        self.path = path
        self.num_x_grid = num_x_grid
        self.use_aux = use_aux
        self.num_lanes = num_lanes
        self.resized_width = resized_width
        self.resized_height = resized_height

        with open(list_path, 'r') as f:
            self.list = f.readlines()

        self.row_anchor = row_anchor
        self.row_anchor.sort()

    def __getitem__(self, index):
        # 获取第 index 个 label 对应的文件名
        l = self.list[index]
        l_info = l.split()
        img_name, label_name = l_info[0], l_info[1]   

        # 从文件名中读取对应的通用特征: 状态,光线,天气
        panaro_label_str = label_name[:-4][-4:]
        status_label = int(panaro_label_str[0])
        lighting_label = int(panaro_label_str[1])
        weather_label = int(panaro_label_str[2])
        road_construction_label = int(panaro_label_str[3])
       
        label_path = label_name
        label = cv2.imread(label_path) # 通道顺序是 (lane_label, type_label, func_color_label)
        img_path = img_name
        img = cv2.imread(img_path) # bgr 顺序读入
        img_ori = np.copy(img)
        if(img is None or label is None):
            print("发现空数据,跳过:", label_path, img_path)
            return None
        seg_label = cv2.split(label)[0]  # 这里使用第一个通道的 lane_label 这个通道作为分割的标签

        # 预处理输入图像------------------
        # 数据增强
        if(self.augment):
            img = augment_hsv(img)  # 处理 bgr 
            img = hist_equalize(img) # 处理 bgr
            img = cutout(img) 
            img, label = cutoutBackground(img, label)
            img, label = random_perspective(img, label)
            # 这里出来的 img 是 bgr 顺序
            # label 出来的顺序是 (lane_label, type_label, func_color_label)

        # if(status_label == 0):
        #     cv2.putText(img, "normal", (100, 200), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5)
        # else:
        #     cv2.putText(img, "abnormal", (100, 200), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5)
        # cv2.imwrite("./train_data_cache/" + str(index) + "_img.png", img)
        # cv2.imwrite("./train_data_cache/" + str(index) + "_label.png", label)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 由于预训练模型 resnet 使用 rgb 图像进行训练的,所以要转换一下通道顺序
        img = cv2.resize(img, (self.resized_width, self.resized_height), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = img.transpose(2, 0, 1) # 把通道数放前面
        img = np.ascontiguousarray(img)
        
        # 预处理标签图像----------------
        label = cv2.resize(label, (self.resized_width, self.resized_height), interpolation=cv2.INTER_NEAREST)

        # 预处理分割标签图像-------------
        if self.use_aux:
            seg_label = cv2.resize(seg_label, (100, 60), interpolation=cv2.INTER_NEAREST)

        # 从标签图像中获取信息------------------------
        # 获取四个标签向量
        lane_pts, pts_type, pts_func, pts_color = self._get_index(label)
        # 将 x 坐标值转换为 one-hot 向量
        cls_label = self._grid_pts(lane_pts, self.num_x_grid, self.resized_width)
        
        if self.use_aux:
            return torch.from_numpy(img).float(), status_label, lighting_label, weather_label, road_construction_label, cls_label, pts_type, pts_func, pts_color, seg_label
        else:
            return torch.from_numpy(img).float(), status_label, lighting_label, weather_label, road_construction_label, cls_label, pts_type, pts_func, pts_color

    def __len__(self):
        return len(self.list)

    def _grid_pts(self, pts, num_cols, w):
        # pts : numlane,n,2
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)

        assert n2 == 2
        to_pts = np.zeros((n, num_lane))
        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray([int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        
        return to_pts.astype(int)

    def _get_index(self, label):
        # 取出标签的第三个通道 
        x_value_label = cv2.split(label)[0] 
        type_label = cv2.split(label)[1]  
        func_color_label = cv2.split(label)[2] 

        # resize 到指定尺寸
        h, w = x_value_label.shape
        # 将 y 轴方向的 anchor 点的坐标映射到原图中的尺寸
        scale_f = lambda x : int((x * 1.0/self.resized_height) * h)
        ori_row_anchors = list(map(scale_f, self.row_anchor))
        num_grid_y_axis = len(ori_row_anchors)

        all_idx_x_value = np.zeros((self.num_lanes, num_grid_y_axis, 2))
        all_idx_type = np.zeros((num_grid_y_axis, self.num_lanes))
        all_idx_func = np.zeros((num_grid_y_axis, self.num_lanes))
        all_idx_color = np.zeros((num_grid_y_axis, self.num_lanes))

        # 遍历原图上面所有的 y 轴坐标值
        for i, curr_ori_y_value in enumerate(ori_row_anchors):
            # 获取当前行的位置坐标值
            curr_row_x_value_label = x_value_label[int(round(curr_ori_y_value))]
            # 遍历所有线
            for lane_idx in range(1, self.num_lanes + 1): # 这里从 1 开始, 因为特征图是从 1 开始
                # 找到当前线对应的 x 坐标值
                curr_lane_pos = np.where(curr_row_x_value_label == lane_idx)[0]
                # 如果未找到当前线 
                if len(curr_lane_pos) == 0:
                    all_idx_x_value[lane_idx - 1, i, 0] = curr_ori_y_value
                    all_idx_x_value[lane_idx - 1, i, 1] = -1
                    all_idx_type[i, lane_idx - 1]  = 0 # 0 代表 线不存在                                        
                    all_idx_func[i, lane_idx - 1]  = 0                                        
                    all_idx_color[i, lane_idx - 1] = 0
                    continue
                
                curr_lane_pos = np.mean(curr_lane_pos) # 取平均是为了取线的中点作为真实位置
                all_idx_x_value[lane_idx - 1, i, 0] = curr_ori_y_value
                all_idx_x_value[lane_idx - 1, i, 1] = curr_lane_pos
                all_idx_type[i, lane_idx - 1]  = type_label[curr_ori_y_value, int(curr_lane_pos)] 
                all_idx_func[i, lane_idx - 1]  = func_color_label[curr_ori_y_value, int(curr_lane_pos)] // 10 
                all_idx_color[i, lane_idx - 1] = func_color_label[curr_ori_y_value, int(curr_lane_pos)] % 10 

        # 数据增强: 将线拓展到图像底端
        all_idx_x_value_cp = all_idx_x_value.copy()
        for i in range(self.num_lanes):
            # 如果一条线所有的点都无效,跳过不拓展
            if np.all(all_idx_x_value_cp[i,:,1] == -1):
                continue
            
            # 找出当前线有效的点的索引
            valid = all_idx_x_value_cp[i, :, 1] != -1
            valid_idx = all_idx_x_value_cp[i, valid, :]
            # 如果所有的点都有效,说明线已经触及到图像底端,无需拓展
            if valid_idx[-1, 0] == all_idx_x_value_cp[0, -1, 0]:
                continue
                
            # 如果一条线有效的点的个数少于 6 个,说明太短了,不拓展
            if len(valid_idx) < 6:
                continue
            
            # 用一半的点进行多项式拟合
            valid_idx_half = valid_idx[len(valid_idx) // 2:, :]
            p = np.polyfit(valid_idx_half[:,0], valid_idx_half[:,1], deg = 1)
            start_line = valid_idx_half[-1,0]
            pos = find_start_pos(all_idx_x_value_cp[i,:,0], start_line) + 1
            
            # 把剩下一半的点都填满
            fitted = np.polyval(p, all_idx_x_value_cp[i,pos:,0])
            fitted = np.array([-1  if y < 0 or y > w-1 else y for y in fitted])

            assert np.all(all_idx_x_value_cp[i,pos:,1] == -1)
            all_idx_x_value_cp[i,pos:,1] = fitted

        if -1 in all_idx_x_value[:, :, 0]:
            pdb.set_trace()

        return all_idx_x_value_cp, all_idx_type, all_idx_func, all_idx_color


def augment_hsv(img, hgain=0.2, sgain=0.2, vgain=0.2):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    return img


def hist_equalize(img, clahe=True, bgr=True, prob=0.5):
    if random.random() < prob:
        # Equalize histogram on BGR image 'img' with img.shape(n,m,3) and range 0-255
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
        if clahe:
            c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            yuv[:, :, 0] = c.apply(yuv[:, :, 0])
        else:
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram\

        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB
    else:
        return img


def flip_horizontally(img, label, prob=0.5):
    if random.random() < prob:
        return cv2.flip(img, 1), cv2.flip(label, 1)
    else:
        return img, label


def random_perspective(img, label, degrees=2, translate=0.0, scale=0.05, shear=5, perspective=0.00001):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] 
    width = img.shape[1] 

    # Center
    C = np.eye(3)
    C[0, 2] = - width / 2  # x translation (pixels)
    C[1, 2] = - height / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1.1 + scale) 
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            label = cv2.warpPerspective(label, M, dsize=(width, height), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))
        else:  
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            label = cv2.warpAffine(label, M[:2], dsize=(width, height), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))

    return img, label


def cutout(image):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = int(random.randint(1, int(h * s)) * 0.7)
        mask_w = int(random.randint(1, int(w * s)) * 0.7)

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]
        
    return image


def cutoutBackground(image, label, prob=0.5):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    if random.random() < prob:
        image[0:int(0.11*h), 0:w-1] = [random.randint(64, 191) for _ in range(3)]
        label[0:int(0.11*h), 0:w-1] = [0 for _ in range(3)]

    return image, label
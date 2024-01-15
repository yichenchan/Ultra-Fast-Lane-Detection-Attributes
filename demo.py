import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
from data.constant import culane_row_anchor, tusimple_row_anchor
import argparse


status_list = ['normal','abnormal']
lighting_list = ['day', 'night', 'others']
weather_list = ['sunny', 'rain', 'snow', 'cloudy', 'unknown']
environmet_list = ['internal', 'closed', 'urban', 'tunnle', 'others']

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    net = parsingNet(pretrained = False, backbone=cfg.backbone, objectness_dim = (cfg.num_x_grid + 1, cfg.num_y_grid, cfg.num_lanes),
                    use_aux=False).cuda() 

    state_dict = torch.load(args.demo_weights, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_w, img_h = cfg.ori_img_width, cfg.ori_img_height
    row_anchor = culane_row_anchor

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vout = cv2.VideoWriter("./out/demo_result.avi", fourcc , 5.0, (img_w, img_h))

    if(args.demo_path.split('/')[-1].split('.')[1].lower() in ['mp4', 'avi', 'mov', 'mkv']):
        demo_video_capture = cv2.VideoCapture(args.demo_path)
        total_frames = int(demo_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    elif(args.demo_path.split('/')[-1].split('.')[1].lower() in ['txt']):
        image_paths = []
        with open(args.demo_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                jpg_path, _ = line.strip().split("  ")
                image_paths.append(jpg_path)
            total_frames = len(image_paths)
    else:
        image_paths = [os.path.join(args.demo_path, file) for file in os.listdir(args.demo_path) if file.endswith(('.jpg', '.jpeg', '.png'))]
        total_frames = len(image_paths)

    frame_index = 0
    while True:
        print("processing frame:", frame_index, "/", total_frames)

        if(args.demo_path.split('/')[-1].split('.')[1].lower() in ['mp4', 'avi', 'mov', 'mkv']):
            ret, img_ori = demo_video_capture.read()
            name = os.path.basename(args.demo_path[:-4] + '_' + str(frame_index) + '.jpg')

            if not ret:
                break
        else:
            if(frame_index >= total_frames):
                break
            image_path = image_paths[frame_index]
            img_ori = cv2.imread(image_path)
            name = os.path.basename(image_path)

        img = img_ori.copy()
        img = cv2.resize(img, (cfg.resized_width, cfg.resized_height), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float().cuda()
        img = torch.unsqueeze(img, 0)
        
        with torch.no_grad():
            out = net(img)

        col_sample = np.linspace(0, cfg.resized_width - 1, cfg.num_x_grid)
        col_sample_w = col_sample[1] - col_sample[0]
        
        status = np.argmax(out[4].data.cpu().numpy(),axis=1)[0]
        lighting = np.argmax(out[5].data.cpu().numpy(),axis=1)[0]
        weather = np.argmax(out[6].data.cpu().numpy(),axis=1)[0]
        env = np.argmax(out[7].data.cpu().numpy(),axis=1)[0]

        objectness_out_j = out[0].data.cpu().numpy()[0,:,:,:]
        objectness_out_j = objectness_out_j[:, ::-1, :]

        prob = scipy.special.softmax(objectness_out_j[:-1, :, :], axis=0)
        idx = np.arange(cfg.num_x_grid) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)

        objectness_out_j = np.argmax(objectness_out_j, axis=0)
        loc[objectness_out_j == cfg.num_x_grid] = 0 # 注意！！！当置信度最大的index是cfg.num_x_grid时，说明是background，未检测到该车道线，直接置 0 
        objectness_out_j = loc
        
        color_out_j = out[1].data.cpu().numpy()[0,:,:,:]
        color_out_j = color_out_j[:, ::-1, :]

        func_out_j = out[2].data.cpu().numpy()[0,:,:,:]
        func_out_j = func_out_j[:, ::-1, :]

        type_out_j = out[3].data.cpu().numpy()[0,:,:,:]
        type_out_j = type_out_j[:, ::-1, :]

        # 遍历每条车道线
        for i in range(objectness_out_j.shape[1]):
            # 当纵向的 grid 检测到车道线的点数超过 0.15 倍的总个数时才有效，否则认为线太短，无效
            if np.sum(objectness_out_j[:, i] != 0) > 0.15 * cfg.num_y_grid:
                # 遍历纵向每一个 grid
                for k in range(objectness_out_j.shape[0]):
                    # 当这个点检测到有车道线时
                    if objectness_out_j[k, i] > 0:
                        # 取出纵向这个点的对应的 color、func、type 输出结果
                        max_color_index = np.argmax(color_out_j[:, k, i])
                        max_func_index = np.argmax(func_out_j[:, k, i])
                        max_type_index = np.argmax(type_out_j[:, k, i])   

                        # 获取该 grid 的中心点画圆
                        ppp = (int(objectness_out_j[k, i] * col_sample_w * img_ori.cols / cfg.resized_width) - 1, int(img_ori.rows * (row_anchor[cfg.num_y_grid-1-k]/cfg.resized_height)) - 1 )

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
                        

        cv2.putText(img_ori, status_list[status], (100, 200), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5)
        cv2.putText(img_ori, lighting_list[lighting], (100, 300), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5)
        cv2.putText(img_ori, weather_list[weather], (100, 400), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5)
        cv2.putText(img_ori, environmet_list[env], (100, 500), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5)
        cv2.imwrite("./out/" + name, img_ori)
        vout.write(img_ori)
        
        vout.release()

        frame_index += 1
import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.constant import culane_row_anchor, tusimple_row_anchor
from PIL import Image
import argparse
from utils.config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--weight', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--config', type=str)
opt = parser.parse_args()
cfg = Config.fromfile(opt.config)

dist_print('start exporting...')
assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

net = parsingNet(pretrained = False, backbone=cfg.backbone, objectness_dim = (cfg.num_x_grid + 1, cfg.num_y_grid, cfg.num_lanes),
                use_aux=False)

state_dict = torch.load(opt.weight, map_location='cpu')['model']
compatible_state_dict = {}
for k, v in state_dict.items():
    if 'module.' in k:
        compatible_state_dict[k[7:]] = v
    else:
        compatible_state_dict[k] = v

net.load_state_dict(compatible_state_dict, strict=False)
net.eval()

img = torch.zeros(1, 3, cfg.resized_height, cfg.resized_width)  
y = net(img)  
ts = torch.jit.trace(net, img)

input_names = ["input"]
output_names = ["x_values", "color_cls", "func_cls", "type_cls", "status_cls", "lighting_cls", "weather_cls", "env_cls"]

torch.onnx.export(net, img, opt.output, opset_version=12, input_names=input_names, output_names=output_names)
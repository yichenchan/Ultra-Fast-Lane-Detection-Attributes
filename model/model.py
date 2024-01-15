import torch
from model.backbone import resnet
import numpy as np

class conv_bn_relu(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size, 
            stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class parsingNet(torch.nn.Module):
    def __init__(self, size=(480, 800), pretrained=True, backbone='50', objectness_dim=(37, 10, 4), use_aux=False):
        super(parsingNet, self).__init__()

        self.size = size
        self.w = size[0]
        self.h = size[1]
        self.objectness_dim = objectness_dim # (num_x_grid, num_y_grid, num_of_lanes)
        self.color_dim = (6, objectness_dim[1], objectness_dim[2]) # (num_color_cls + 'background', num_y_grid, num_of_lanes)
        self.func_dim =  (8, objectness_dim[1], objectness_dim[2]) # (num_func_cls + 'background', num_y_grid, num_of_lanes)
        self.type_dim =  (9, objectness_dim[1], objectness_dim[2]) # (num_type_cls + 'background', num_y_grid, num_of_lanes)
        self.use_aux = use_aux

        self.model = resnet(backbone, pretrained=pretrained)

        # 全局状态输出头, 用来降特征图数量减少全链接的参数量
        # self.global_cls_pool = torch.nn.Conv2d(self.model.outDim4, self.model.outDim1, 1) 
        self.global_cls_pool = torch.nn.Conv2d(self.model.outDim4, 8, 1) 
        # self.global_cls_pool = torch.nn.Sequential(
        #     torch.nn.Conv2d(self.model.outDim4, self.model.outDim3, 3, 1, 0),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(self.model.outDim3, self.model.outDim2, 3, 1, 0),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(self.model.outDim2, self.model.outDim1, 3, 1, 0),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(self.model.outDim1, 8, 3, 1, 0)
        # )

        # 状态向量输出头
        self.status_cls = torch.nn.Sequential(
            torch.nn.LazyLinear(4),
            torch.nn.BatchNorm1d(4),
            # torch.nn.ReLU(),
            torch.nn.Linear(4, 2), # status_list = ['normal','abnormal']
        )

        # 白天夜晚输出头
        self.lighting_cls = torch.nn.Sequential(
            torch.nn.LazyLinear(16),
            torch.nn.BatchNorm1d(16),
            # torch.nn.ReLU(),
            torch.nn.Linear(16, 3), # lighting_list = ['day', 'night', 'others']
        )

        # 天气向量输出头
        self.weather_cls = torch.nn.Sequential(
            torch.nn.LazyLinear(16),
            torch.nn.BatchNorm1d(16),
            # torch.nn.ReLU(),
            torch.nn.Linear(16, 5), #weather_list = ['sunny', 'rain', 'snow', 'cloudy', 'unknown']
        )

        # 环境状态向量输出头
        self.env_cls = torch.nn.Sequential(
            torch.nn.LazyLinear(16),
            torch.nn.BatchNorm1d(16),
            # torch.nn.ReLU(),
            torch.nn.Linear(16, 5), #environmet_list = ['internal', 'closed', 'urban', 'tunnle', 'others']
        )

        # 线的特征输出头池化
        self.lane_feature_pool = torch.nn.Conv2d(self.model.outDim4, self.model.outDim1, 1) 
        # self.lane_feature_pool = torch.nn.Sequential(
        #     torch.nn.Conv2d(self.model.outDim4, self.model.outDim3, 3, 1, 0),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(self.model.outDim3, self.model.outDim2, 3, 1,0),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(self.model.outDim2, self.model.outDim1, 3, 1,0),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(self.model.outDim1, 8, 3, 1, 0)
        # )

        # 线的类别向量输出头
        self.x_value = torch.nn.Sequential(
            torch.nn.LazyLinear(2048),
            torch.nn.BatchNorm1d(2048),
            # torch.nn.ReLU(),
            torch.nn.Linear(2048, np.prod(self.objectness_dim)), # 输出所有 12 条车道线在每个 grid 是否存在 (num_x_grid, num_y_grid, num_of_lanes)
        )

        # 线的颜色向量输出头
        self.color_cls = torch.nn.Sequential(
            torch.nn.LazyLinear(64),
            torch.nn.BatchNorm1d(64),
            # torch.nn.ReLU(),
            torch.nn.Linear(64, np.prod(self.color_dim)), # 输出所有 12 条车道线在 y 轴上不同分段的颜色类别 (6, num_y_grid, num_of_lanes)
        )

        # 线的功能向量输出头
        self.func_cls = torch.nn.Sequential(
            torch.nn.LazyLinear(64),
            torch.nn.BatchNorm1d(64),
            # torch.nn.ReLU(),
            torch.nn.Linear(64, np.prod(self.func_dim)), # 输出所有 12 条车道线在 y 轴上不同分段的功能类别 (8, num_y_grid, num_of_lanes)
        )

        # 线的类别向量输出头
        self.type_cls = torch.nn.Sequential(
            torch.nn.LazyLinear(64),
            torch.nn.BatchNorm1d(64),
            # torch.nn.ReLU(),
            torch.nn.Linear(64, np.prod(self.type_dim)),# 输出所有 12 条车道线在 y 轴上不同分段的type类别 (9, num_y_grid, num_of_lanes)
        )

        if self.use_aux:
            self.aux_header2 = torch.nn.Sequential(
                conv_bn_relu(self.model.outDim2, self.model.outDim2, 3, padding=1),
                conv_bn_relu(self.model.outDim2, self.model.outDim2, 3, padding=1),
                conv_bn_relu(self.model.outDim2, self.model.outDim2, 3, padding=1),
                conv_bn_relu(self.model.outDim2, self.model.outDim2, 3, padding=1),
            )
            self.aux_header3 = torch.nn.Sequential(
                conv_bn_relu(self.model.outDim3, self.model.outDim2, 3, padding=1),
                conv_bn_relu(self.model.outDim2, self.model.outDim2, 3, padding=1),
                conv_bn_relu(self.model.outDim2, self.model.outDim2, 3, padding=1),
            )
            self.aux_header4 = torch.nn.Sequential(
                conv_bn_relu(self.model.outDim4, self.model.outDim2, 3, padding=1),
                conv_bn_relu(self.model.outDim2, self.model.outDim2, 3, padding=1),
            )
            self.aux_combine = torch.nn.Sequential(
                conv_bn_relu(self.model.outDim2 * 3, self.model.outDim3, 3, padding=2, dilation=2),
                conv_bn_relu(self.model.outDim3, self.model.outDim2, 3, padding=2, dilation=2),
                conv_bn_relu(self.model.outDim2, self.model.outDim2, 3, padding=2, dilation=2),
                conv_bn_relu(self.model.outDim2, self.model.outDim2, 3, padding=4, dilation=4),
                torch.nn.Conv2d(self.model.outDim2, objectness_dim[-1] + 1, 1)
                # output : n, num_of_lanes+1, h, w
            )
            initialize_weights(self.aux_header2, self.aux_header3, self.aux_header4, self.aux_combine)

        # 初始化参数
        initialize_weights(self.global_cls_pool)
        initialize_weights(self.lane_feature_pool)
        initialize_weights(self.x_value)
        initialize_weights(self.color_cls)
        initialize_weights(self.func_cls)
        initialize_weights(self.type_cls)
        initialize_weights(self.status_cls)
        initialize_weights(self.lighting_cls)
        initialize_weights(self.weather_cls)
        initialize_weights(self.env_cls)

    def forward(self, x):
        # backbone 提特征
        x2, x3, x4 = self.model(x)

        # 全局特征从 x4 进行降维
        global_cls_head = self.global_cls_pool(x4)    
        # 全局特征图 flat 成二维向量
        global_cls_head = global_cls_head.view(global_cls_head.size(0), -1)

        # 全连接输出四个状态向量
        status_cls = self.status_cls(global_cls_head).view(-1, 2)
        lighting_cls = self.lighting_cls(global_cls_head).view(-1, 3)
        weather_cls = self.weather_cls(global_cls_head).view(-1, 5)
        env_cls = self.env_cls(global_cls_head).view(-1, 5)

        # 线的特征从 x4 进行降维
        lane_feature_head = self.lane_feature_pool(x4)
        lane_feature_head = lane_feature_head.view(lane_feature_head.size(0), -1)

        # 全连接输出四个线的特征向量
        x_values = self.x_value(lane_feature_head).view(-1, *self.objectness_dim) # [1, 201, 18, 12]
        color_cls = self.color_cls(lane_feature_head).view(-1, *self.color_dim) # [1, 6, 18, 12]
        func_cls = self.func_cls(lane_feature_head).view(-1, *self.func_dim) # [1, 8, 18, 12]
        type_cls = self.type_cls(lane_feature_head).view(-1, *self.type_dim) # [1, 9, 18, 12]

        # 不同层输出附属分割头最后合并,用来进行线的分割,进行辅助的特征提取
        if self.use_aux and self.training:
            aux_x2 = self.aux_header2(x2)
            aux_x3 = self.aux_header3(x3)
            aux_x3 = torch.nn.functional.interpolate(aux_x3, scale_factor = 2, mode='bilinear')
            aux_x4 = self.aux_header4(x4)
            aux_x4 = torch.nn.functional.interpolate(aux_x4, scale_factor = 4, mode='bilinear')
            aux_seg = torch.cat([aux_x2, aux_x3, aux_x4], dim=1)
            aux_seg = self.aux_combine(aux_seg)
            
            return x_values, color_cls, func_cls, type_cls, status_cls, lighting_cls, weather_cls, env_cls, aux_seg

        return x_values, color_cls, func_cls, type_cls, status_cls, lighting_cls, weather_cls, env_cls


def initialize_weights(*models):
    for model in models:
        real_init_weights(model)

def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):    
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)
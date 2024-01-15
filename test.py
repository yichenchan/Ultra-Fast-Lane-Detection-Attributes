import torch, os
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
from evaluation.eval_wrapper import eval_lane
import torch
from data.dataloader import get_dataloader
from utils.dist_utils import dist_print, dist_tqdm, is_main_process, DistSummaryWriter
import scipy

eps = 0.0000001

def test(net, test_loader, cfg, local_rank):
    net.eval()

    x_pos_correct_total = [0] * 12
    x_pos_total = [0] * 12
    x_pos_tp_total = [0] * 12
    x_pos_p_total = [0] * 12
    color_correct_total = [0] * 12
    color_total = [0] * 12
    color_tp_total = [0] * 12
    color_p_total = [0] * 12
    type_correct_total = [0] * 12
    type_total = [0] * 12
    type_tp_total = [0] * 12
    type_p_total = [0] * 12
    func_correct_total = [0] * 12
    func_total = [0] * 12
    func_tp_total = [0] * 12
    func_p_total = [0] * 12
    img_num_total = 0
    status_correct_total = 0
    status_tp_total = 0
    status_p_total = 0
    lighting_correct_total = 0
    lighting_tp_total = 0
    lighting_p_total = 0
    weather_correct_total = 0
    weather_tp_total = 0
    weather_p_total = 0    
    road_construction_correct_total = 0
    road_construction_tp_total = 0
    road_construction_p_total = 0
    for i, data in enumerate(dist_tqdm(test_loader)):
        # 获取标签
        img, status_label, lighting_label, weather_label, road_construction_label, cls_label, type_label, func_label, color_label = data
        img, status_label, lighting_label, weather_label, road_construction_label, cls_label_, type_label_, func_label_, color_label_ = \
        img.cuda(), status_label.cuda(), lighting_label.cuda(), weather_label.cuda(), road_construction_label.cuda(), cls_label.cuda(), type_label.cuda(), func_label.cuda(), color_label.cuda()

        # 获取预测输出
        cls_out, color_out, func_out, type_out, status_out, lighting_out, weather_out, env_out = net(img)
        
        # 获取状态输出
        status_out_prob = torch.nn.functional.softmax(status_out, dim=1)
        status_out_max_index = torch.argmax(status_out_prob, dim=1)
        status_correct_total += (status_out_max_index == status_label).sum().item()
        status_tp_total += ((status_out_max_index == status_label) & (status_label == 0)).sum().item()
        status_p_total += (status_label == 0).sum().item()

        # 获取时间输出
        lighting_out_prob = torch.nn.functional.softmax(lighting_out, dim=1)
        lighting_out_max_index = torch.argmax(lighting_out_prob, dim=1)
        lighting_correct_total += (lighting_out_max_index == lighting_label).sum().item()
        lighting_tp_total += ((lighting_out_max_index == lighting_label) & (lighting_label != 2)).sum().item()
        lighting_p_total += (lighting_label != 2).sum().item()

        # 获取天气输出
        weather_out_prob = torch.nn.functional.softmax(weather_out, dim=1)
        weather_out_max_index = torch.argmax(weather_out_prob, dim=1)
        weather_correct_total += (weather_out_max_index == weather_label).sum().item()
        weather_tp_total += ((weather_out_max_index == weather_label) & (weather_label != 4)).sum().item()
        weather_p_total += (weather_label != 4).sum().item()

        # 获取道路情况输出
        env_out_prob = torch.nn.functional.softmax(env_out, dim=1)
        env_out_max_index = torch.argmax(env_out_prob, dim=1)
        road_construction_correct_total += (env_out_max_index == road_construction_label).sum().item()
        road_construction_tp_total += ((env_out_max_index == road_construction_label) & (road_construction_label != 4)).sum().item()
        road_construction_p_total += (road_construction_label != 4).sum().item()

        # 过滤所有正常图片的 mask(非正常状态不进行测试)
        normal_status_mask = (status_label == 0)
        img_num_total += img.shape[0]

        # 获取道路线 x 坐标输出
        cls_out_prob = torch.nn.functional.softmax(cls_out, dim=1)
        cls_out_max_index_ = torch.argmax(cls_out_prob, dim=1)

        # 获取道路线颜色输出
        color_out_prob = torch.nn.functional.softmax(color_out, dim=1)
        color_out_max_index_ = torch.argmax(color_out_prob, dim=1)

        # 获取道路线类型输出
        type_out_prob = torch.nn.functional.softmax(type_out, dim=1)
        type_out_max_index_ = torch.argmax(type_out_prob, dim=1)

        # 获取道路线功能输出
        func_out_prob = torch.nn.functional.softmax(func_out, dim=1)
        func_out_max_index_ = torch.argmax(func_out_prob, dim=1)

        for i in range(12):
            cls_label = cls_label_[normal_status_mask] # 筛选 normal 的样例
            cls_out_max_index = cls_out_max_index_[normal_status_mask] # 筛选 normal 的样例
            x_pos_correct_total[i] += (cls_out_max_index[:, :, i] == cls_label[:, :, i]).sum().item()
            x_pos_total[i] += cls_label[:, :, i].numel()
            x_pos_tp_total[i] += ((cls_out_max_index[:, :, i] == cls_label[:, :, i]) & (cls_label[:, :, i] != cfg.num_x_grid)).sum().item()
            x_pos_p_total[i] += (cls_label[:, :, i] != cfg.num_x_grid).sum().item()
            
            color_label = color_label_[normal_status_mask] # 筛选 normal 的样例
            color_out_max_index = color_out_max_index_[normal_status_mask] # 筛选 normal 的样例
            color_correct_total[i] += (color_out_max_index[:, :, i] == color_label[:, :, i]).sum().item()
            color_total[i] += color_label[:, :, i].numel()
            color_tp_total[i] += ((color_out_max_index[:, :, i] == color_label[:, :, i]) & (color_label[:, :, i] != 0)).sum().item()
            color_p_total[i] += (color_label[:, :, i] != 0).sum().item()
            
            type_label = type_label_[normal_status_mask] # 筛选 normal 的样例
            type_out_max_index = type_out_max_index_[normal_status_mask] # 筛选 normal 的样例
            type_correct_total[i] += (type_out_max_index[:, :, i] == type_label[:, :, i]).sum().item()
            type_total[i] += type_label[:, :, i].numel()
            type_tp_total[i] += ((type_out_max_index[:, :, i] == type_label[:, :, i]) & (type_label[:, :, i] != 0)).sum().item()
            type_p_total[i] += (type_label[:, :, i] != 0).sum().item()

            func_label = func_label_[normal_status_mask] # 筛选 normal 的样例
            func_out_max_index = func_out_max_index_[normal_status_mask] # 筛选 normal 的样例
            func_correct_total[i] += (func_out_max_index[:, :, i] == func_label[:, :, i]).sum().item()
            func_total[i] += func_label[:, :, i].numel()
            func_tp_total[i] += ((func_out_max_index[:, :, i] == func_label[:, :, i]) & (func_label[:, :, i] != 0)).sum().item()
            func_p_total[i] += (func_label[:, :, i] != 0).sum().item()                        

    if is_main_process():
        print("             precision(total)", "      recall(positive_total)")
        print("status         ", format(status_correct_total / (img_num_total + eps), ".4f"), "(", img_num_total, ")           ", \
                                format(status_tp_total / (status_p_total + eps), ".4f"), "(", status_p_total, ")")
        print("lighting       ", format(lighting_correct_total / (img_num_total + eps), ".4f"), "(", img_num_total, ")           ", \
                                format(lighting_tp_total / (lighting_p_total + eps), ".4f"), "(", lighting_p_total, ")")
        print("weather        ", format(weather_correct_total / (img_num_total + eps), ".4f"), "(", img_num_total, ")           ", \
                                format(weather_tp_total / (weather_p_total + eps), ".4f"), "(", weather_p_total, ")")
        print("urbanOrNot     ", format(road_construction_correct_total / (img_num_total + eps), ".4f"), "(", img_num_total, ")           ", \
                                format(road_construction_tp_total / (road_construction_p_total + eps), ".4f"), "(", road_construction_p_total, ")")            

        for i in range(12):
            print("line", format(i, "02d"), "x      ", format(x_pos_correct_total[i] / (x_pos_total[i] + eps), ".4f"), "(", x_pos_total[i], ")          ", \
                                        format(x_pos_tp_total[i] / (x_pos_p_total[i] + eps), ".4f"), "(", x_pos_p_total[i], ")")
            print("line", format(i, "02d"), "color  ", format(color_correct_total[i] / (color_total[i] + eps), ".4f"), "(", color_total[i], ")          ", \
                                        format(color_tp_total[i] / (color_p_total[i] + eps), ".4f"), "(", color_p_total[i], ")")
            print("line", format(i, "02d"), "type   ", format(type_correct_total[i] / (type_total[i] + eps), ".4f"), "(", type_total[i], ")          ", \
                                        format(type_tp_total[i] / (type_p_total[i] + eps), ".4f"), "(", type_p_total[i], ")")
            print("line", format(i, "02d"), "func   ", format(func_correct_total[i] / (func_total[i] + eps), ".4f"), "(", func_total[i], ")          ", \
                                        format(func_tp_total[i] / (func_p_total[i] + eps), ".4f"), "(", func_p_total[i], ")")                                    


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    distributed = False
    # if 'WORLD_SIZE' in os.environ:
    #     distributed = int(os.environ['WORLD_SIZE']) > 1

    # if distributed:
    #     torch.cuda.set_device(args.local_rank)
    #     torch.distributed.init_process_group(backend='nccl', init_method='env://')

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    net = parsingNet(pretrained = False, backbone=cfg.backbone, objectness_dim = (cfg.num_x_grid+1, cfg.num_y_grid, cfg.num_lanes), use_aux=False).cuda() 

    state_dict = torch.load(cfg.test_model, map_location = 'cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict = False)

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])

    test_loader = get_dataloader(cfg.test_batch_size, cfg.data_root, cfg.num_x_grid, cfg.test_dataset, False, False, distributed, cfg.num_lanes)

    test(net, test_loader, cfg)
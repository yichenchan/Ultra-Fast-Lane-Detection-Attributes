import torch, os, datetime
import numpy as np
# import torch
import torch.nn as nn
from model.model import parsingNet
from data.dataloader import get_dataloader

from utils.dist_utils import dist_print, dist_tqdm, is_main_process, DistSummaryWriter
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import MultiLabelAcc, AccTopk, Metric_mIoU, update_metrics, reset_metrics

from utils.loss import SoftmaxFocalLoss, ParsingRelationLoss, ParsingRelationDis
from utils.metrics import MultiLabelAcc, AccTopk, Metric_mIoU
from utils.dist_utils import DistSummaryWriter

from utils.common import merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger

from evaluation.eval_wrapper import eval_lane
from test import test
import time


def inference(net, data_label, use_aux):
    
    if use_aux:
        img, status_label, lighting_label, weather_label, road_construction_label, cls_label, pts_type, pts_func, pts_color, seg_label = data_label
        img, status_label, lighting_label, weather_label, road_construction_label, cls_label, pts_type, pts_func, pts_color, seg_label = \
        img.cuda(), status_label.cuda(), lighting_label.cuda(), weather_label.cuda(), road_construction_label.cuda(), cls_label.cuda(), pts_type.cuda(), pts_func.cuda(), pts_color.cuda(), seg_label.cuda()
        cls_out, color_out, func_out, type_out, status_out, lighting_out, weather_out, env_out, seg_out = net(img)
        
        return {'cls_out': cls_out, 'cls_label': cls_label, 
                'status_out':status_out, 'status_label': status_label,
                'lighting_out':lighting_out, 'lighting_label': lighting_label,
                'weather_out':weather_out, 'weather_label': weather_label,
                'env_out':env_out, 'env_label': road_construction_label,
                'type_out':type_out, 'type_label': pts_type,
                'func_out':func_out, 'func_label': pts_func,
                'color_out':color_out, 'color_label': pts_color,
                'seg_out':seg_out, 'seg_label': seg_label   
        }

    else:
        img, cls_label = data_label
        img, cls_label = img.cuda(), cls_label.long().cuda()
        cls_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label}


def resolve_val_data(results, use_aux):
    results['cls_out'] = torch.argmax(results['cls_out'], dim=1)
    if use_aux:
        results['seg_out'] = torch.argmax(results['seg_out'], dim=1)
    return results


def calc_loss(loss_dict, results, logger, global_step):
    loss = 0

    loss_status = nn.CrossEntropyLoss()(results['status_out'], results['status_label'])
    loss_lighting = nn.CrossEntropyLoss()(results['lighting_out'], results['lighting_label'])
    loss_weather = nn.CrossEntropyLoss()(results['weather_out'], results['weather_label'])
    loss_env = nn.CrossEntropyLoss()(results['env_out'], results['env_label'])

    cls_label = results['cls_label'][results['status_label'] == 0, :, :]
    cls_out = results['cls_out'][results['status_label'] == 0, :, :, :] # [1, 201, 18, 12]
    loss_cls = SoftmaxFocalLoss(2)(cls_out, cls_label) + 0.0 * ParsingRelationLoss()(cls_out) + 0.0 * ParsingRelationDis()(cls_out)

    pts_type = results['type_label'][results['status_label']==0, :, :]
    type_out = results['type_out'][results['status_label']==0, :, :, :]
    loss_type = SoftmaxFocalLoss(2)(type_out, pts_type.long())

    pts_func = results['func_label'][results['status_label']==0, :, :]
    func_out = results['func_out'][results['status_label']==0, :, :, :]
    loss_func = SoftmaxFocalLoss(2)(func_out, pts_func.long())

    pts_color = results['color_label'][results['status_label']==0, :, :]
    color_out = results['color_out'][results['status_label']==0, :, :, :]
    loss_color = SoftmaxFocalLoss(2)(color_out, pts_color.long())

    seg_label = results['seg_label'][results['status_label']==0, :, :]
    seg_out = results['seg_out'][results['status_label']==0, :, :, :]
    loss_seg = torch.nn.CrossEntropyLoss()(seg_out, seg_label.long())

    loss = (loss_status + loss_lighting + loss_weather + loss_env) * 0.01 + loss_cls  + loss_type + loss_func + loss_color + loss_seg 
    return loss, (loss_status, loss_lighting, loss_weather, loss_env, loss_cls, loss_type, loss_func, loss_color, loss_seg)


def train(net, train_dataloader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, use_aux):
    progress_bar = dist_tqdm(train_dataloader)

    for b_idx, data_label in enumerate(progress_bar):
        global_step = epoch * len(train_dataloader) + b_idx

        # 训练
        net.train()
        reset_metrics(metric_dict)
        results = inference(net, data_label, use_aux)
        loss, sep_loss = calc_loss(loss_dict, results, logger, global_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)
        results = resolve_val_data(results, use_aux)
        update_metrics(metric_dict, results)

        if global_step % 20 == 0:
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
        logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if hasattr(progress_bar,'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
            progress_bar.set_postfix(total = '%.3f' % float(loss),        
                                    status = '%.3f' % sep_loss[0],
                                    light = '%.3f' % sep_loss[1],
                                    weather = '%.3f' % sep_loss[2],
                                    env = '%.3f' % sep_loss[3],
                                    x = '%.3f' % sep_loss[4],
                                    type = '%.3f' % sep_loss[5],
                                    func = '%.3f' % sep_loss[6],
                                    color = '%.3f' % sep_loss[7],
                                    **kwargs)

        t_data_0 = time.time()
        
if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    work_dir = get_work_dir(cfg)

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    # 创建训练数据集
    train_loader = get_dataloader(cfg.train_batch_size, cfg.data_root, cfg.num_x_grid, cfg.train_dataset, cfg.use_augment_in_train, cfg.use_aux, distributed, cfg.num_lanes, cfg.resized_width, cfg.resized_height)

    # 只在第一块 gpu 上进行验证,创建验证集
    train_loader_for_val = get_dataloader(cfg.val_batch_size, cfg.data_root, cfg.num_x_grid, cfg.train_dataset, cfg.use_augment_in_val, False, False, cfg.num_lanes, cfg.resized_width, cfg.resized_height)
    val_loader = get_dataloader(cfg.val_batch_size, cfg.data_root, cfg.num_x_grid, cfg.val_dataset, cfg.use_augment_in_val, False, False, cfg.num_lanes, cfg.resized_width, cfg.resized_height)

    net = parsingNet(pretrained = True, backbone=cfg.backbone, objectness_dim = (cfg.num_x_grid + 1, cfg.num_y_grid, cfg.num_lanes), use_aux=cfg.use_aux).cuda()

    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # only use backbone parameters
        for k,v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
        # 冻结编码器部分
        # for name, param in net.named_parameters():
        #     if 'model' in name:
        #         param.requires_grad = False

    if distributed:
        dummy_input = torch.randn(1, 3, cfg.resized_height, cfg.resized_width).cuda()
        net(dummy_input)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])

    optimizer = get_optimizer(net, cfg)

    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'], strict=False)
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
        work_dir = os.path.dirname(cfg.resume)
    else:
        resume_epoch = 0

    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    dist_print(len(train_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    logger = get_logger(work_dir, cfg)
    cp_projects(args.auto_backup, work_dir)

    for epoch in range(resume_epoch, cfg.epoch):
        # 训练
        train(net, train_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, cfg.use_aux)
        
        # 验证
        # test(net, train_loader_for_val, cfg, args.local_rank)
        test(net, val_loader, cfg, args.local_rank)

        # 保存
        save_model(net, optimizer, epoch , work_dir, distributed)

    # 进行最终测试
    test_loader = get_dataloader(cfg.test_batch_size, cfg.data_root, cfg.num_x_grid, cfg.test_dataset, cfg.use_augment_in_test, False, distributed, cfg.num_lanes, cfg.resized_width, cfg.resized_height)
    test(net, test_loader, cfg, args.local_rank)

    logger.close()

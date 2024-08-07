import os
import time
import yaml
import pprint
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from evaluate import evaluate
from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log
from util.dist_helper import setup_distributed

from util.train_utils import (DictAverageMeter, confidence_weighted_loss,
                               cutmix_img_, cutmix_mask, generate_lambda_schedule)
from prev_utils import update_previous_list, get_previous_logits

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)
    ddp = True if world_size > 1 else False
    amp = cfg['amp']

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    list_max_len = cfg['prev']['list_max_len']
    base_only_epoch = cfg['prev']['base_only_epoch']
    prev_conf_thresh = cfg['prev']['conf_thresh']
    prev_model_num = cfg['prev']['model_num']
    prev_random_select = cfg['prev']['random_select']
    prev_conf_mode = 'pixelavg' if cfg['dataset'] == 'cityscapes' else 'pixelwise'
    conf_thresh = cfg['conf_thresh']

    model = DeepLabV3Plus(cfg)
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    if ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                        output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    if ddp:
        trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
        trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                                   pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
        trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
        trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                                   pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
        valsampler = torch.utils.data.distributed.DistributedSampler(valset)
        valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                               drop_last=False, sampler=valsampler)
    else:
        trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                                pin_memory=True, num_workers=1, shuffle=True, drop_last=True)
        trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                                pin_memory=True, num_workers=1, shuffle=True, drop_last=True)
        valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    total_epochs = cfg['epochs']
    total_iters = len(trainloader_u) * total_epochs
    epoch = -1
    previous_best = 0.0
    ETA = 0.0
    prev_model_list = []
    
    scaler = torch.cuda.amp.GradScaler()
    is_best = False
    for epoch in range(epoch + 1, total_epochs):
        start_time = time.time()

        if is_best and (epoch >= base_only_epoch):
            update_previous_list(prev_model_list, model, list_max_len=list_max_len)

        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}, ETA: {:.2f}M'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best, ETA/60))

        log_avg = DictAverageMeter()

        if ddp:
            trainloader_l.sampler.set_epoch(epoch)
            trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)
        current_lambda = generate_lambda_schedule(epoch, total_epochs, total_epochs // 2)

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):
            
            t0 = time.time()
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()
            iters = epoch * len(trainloader_u) + i

            # CutMix images
            cutmix_img_(img_u_s1, img_u_s1_mix, cutmix_box1)
            cutmix_img_(img_u_s2, img_u_s2_mix, cutmix_box2)

            # Use AMP
            with torch.cuda.amp.autocast(enabled=amp):
                
                model.eval()
                if amp:
                    pred_u_w_mix = model(img_u_w_mix).detach()
                    conf_u_w_mix, mask_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)
                else:
                    with torch.no_grad():
                        pred_u_w_mix = model(img_u_w_mix).detach()
                        conf_u_w_mix, mask_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)

                # Generate previous guidance
                if len(prev_model_list) != 0:
                    pred_u_w_prev = get_previous_logits(prev_model_list, torch.cat((img_u_w, img_u_w_mix)), prev_model_num, prev_random_select)
                    pred_u_w_prev, pred_u_w_mix_prev = pred_u_w_prev.chunk(2)
                    
                    conf_u_w_prev, mask_u_w_prev = pred_u_w_prev.softmax(dim=1).max(dim=1)
                    conf_u_w_mix_prev, mask_u_w_mix_prev = pred_u_w_mix_prev.softmax(dim=1).max(dim=1)

                model.train()

                num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

                preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
                pred_x, pred_u_w = preds.split([num_lb, num_ulb])
                pred_u_w_fp = preds_fp[num_lb:]

                pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

                pred_u_w = pred_u_w.detach()
                conf_u_w, mask_u_w = pred_u_w.softmax(dim=1).max(dim=1)

                # CutMix labels
                mask_u_w_cutmixed1 = cutmix_mask(mask_u_w, mask_u_w_mix, cutmix_box1)
                mask_u_w_cutmixed2 = cutmix_mask(mask_u_w, mask_u_w_mix, cutmix_box2)
                conf_u_w_cutmixed1 = cutmix_mask(conf_u_w, conf_u_w_mix, cutmix_box1)
                conf_u_w_cutmixed2 = cutmix_mask(conf_u_w, conf_u_w_mix, cutmix_box2)
                ignore_mask_cutmixed1 = cutmix_mask(ignore_mask, ignore_mask_mix, cutmix_box1)
                ignore_mask_cutmixed2 = cutmix_mask(ignore_mask, ignore_mask_mix, cutmix_box2)
                
                # Generate previous guidance
                if len(prev_model_list) != 0:
                    mask_u_w_cutmixed1_prev = cutmix_mask(mask_u_w_prev, mask_u_w_mix_prev, cutmix_box1)
                    mask_u_w_cutmixed2_prev = cutmix_mask(mask_u_w_prev, mask_u_w_mix_prev, cutmix_box2)
                    conf_u_w_cutmixed1_prev = cutmix_mask(conf_u_w_prev, conf_u_w_mix_prev, cutmix_box1)
                    conf_u_w_cutmixed2_prev = cutmix_mask(conf_u_w_prev, conf_u_w_mix_prev, cutmix_box2)

                loss_x = criterion_l(pred_x, mask_x)
                
                loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
                loss_u_s1 = confidence_weighted_loss(loss_u_s1, conf_u_w_cutmixed1, ignore_mask_cutmixed1, conf_thresh=conf_thresh)
                
                loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
                loss_u_s2 = confidence_weighted_loss(loss_u_s2, conf_u_w_cutmixed2, ignore_mask_cutmixed2, conf_thresh=conf_thresh)

                loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
                loss_u_w_fp = confidence_weighted_loss(loss_u_w_fp, conf_u_w, ignore_mask, conf_thresh=conf_thresh)

                mask_ratio = ((conf_u_w >= conf_thresh) & (ignore_mask != 255)).sum().item() / \
                    (ignore_mask != 255).sum()

                loss_standard = loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5

                if len(prev_model_list) != 0:
                    # For PrevMatch Losses
                    loss_u_s1_prev = criterion_u(pred_u_s1, mask_u_w_cutmixed1_prev)
                    loss_u_s1_prev = confidence_weighted_loss(loss_u_s1_prev, conf_u_w_cutmixed1_prev, ignore_mask_cutmixed1, conf_thresh=prev_conf_thresh, conf_mode=prev_conf_mode)
                    
                    loss_u_s2_prev = criterion_u(pred_u_s2, mask_u_w_cutmixed2_prev)
                    loss_u_s2_prev = confidence_weighted_loss(loss_u_s2_prev, conf_u_w_cutmixed2_prev, ignore_mask_cutmixed2, conf_thresh=prev_conf_thresh, conf_mode=prev_conf_mode)

                    loss_u_w_fp_prev = criterion_u(pred_u_w_fp, mask_u_w_prev)
                    loss_u_w_fp_prev = confidence_weighted_loss(loss_u_w_fp_prev, conf_u_w_prev, ignore_mask, conf_thresh=prev_conf_thresh, conf_mode=prev_conf_mode)

                    mask_ratio_prev = ((conf_u_w_prev >= prev_conf_thresh) & (ignore_mask != 255)).sum().item() / \
                        (ignore_mask != 255).sum()
                    
                    loss_prev = (loss_u_s1_prev * 0.25 + loss_u_s2_prev * 0.25 + loss_u_w_fp_prev * 0.5) * current_lambda
                    total_loss = (loss_x + loss_standard + loss_prev) / (2.0 + current_lambda)
                else:
                    total_loss = (loss_x + loss_standard) / 2.0

            if ddp:
                torch.distributed.barrier()

            optimizer.zero_grad()
            if amp:
                loss = scaler.scale(total_loss)
                loss.backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            # Logging
            log_avg.update({
                'iter time': time.time() - t0,
                'Total loss': total_loss,          # Logging original loss (=total loss), not AMP scaled loss
                'Loss x': loss_x,
                'Loss s': (loss_u_s1 + loss_u_s2) / 2.0,
                'Loss w_fp': loss_u_w_fp,
                'Mask ratio': mask_ratio,
            })
            
            # Logging for PrevMatch
            if len(prev_model_list) != 0:
                log_avg.update({
                'Loss s_prev': (loss_u_s1_prev + loss_u_s2_prev) / 2.0,
                'Loss w_fp_prev': loss_u_w_fp_prev,
                'Mask ratio_prev': mask_ratio_prev,
                })

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            if rank == 0:
                for k, v in log_avg.avgs.items():
                    writer.add_scalar('train/'+k, v, iters)
            
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(f'Iters: {i}, ' + str(log_avg))
                log_avg.reset()

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, args.save_path, ddp)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))

        end_time = time.time()
        time_per_epoch = end_time - start_time
        ETA = (total_epochs - (epoch + 1)) * time_per_epoch

if __name__ == '__main__':
    main()

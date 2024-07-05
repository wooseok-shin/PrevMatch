import os
import yaml
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.classes import CLASSES
from util.utils import count_params, AverageMeter, intersectionAndUnion, color_map

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from saved weights in multi-GPU training"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_key = k[len('module.'):]
        else:
            new_key = k
        new_state_dict[new_key] = v

    return new_state_dict

def evaluate(model, loader, mode, cfg, save_path, ddp=False):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    palette = color_map(cfg['dataset'])
    
    if cfg['save_map']:
        save_path = save_path.split('.pth')[0]    # checkpoints/pascal_92_73.7.pth -> checkpoints/pascal_92_73.7
        os.makedirs(os.path.join(save_path, 'mask'), exist_ok=True)           # for mask
        os.makedirs(os.path.join(save_path, 'color_mask'), exist_ok=True)   # for colorized mask

    with torch.no_grad():
        for img, mask, id in tqdm(loader):

            img = img.cuda()

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                pred = model(img).argmax(dim=1)

            # Save prediction mask
            if cfg['save_map']:
                img_path, mask_path = id[0].split(' ')
                mask_name = os.path.basename(mask_path)
                
                pred_map = pred[0].cpu().numpy().astype(np.uint8)
                pred_map = Image.fromarray(pred_map)
                pred_colormap = pred_map.convert('P')
                pred_colormap.putpalette(palette)
                
                pred_map.save(os.path.join(save_path, 'mask', mask_name))
                pred_colormap.save(os.path.join(save_path, 'color_mask', mask_name))

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            if ddp:
                reduced_intersection = torch.from_numpy(intersection).cuda()
                reduced_union = torch.from_numpy(union).cuda()
                reduced_target = torch.from_numpy(target).cuda()

                dist.all_reduce(reduced_intersection)
                dist.all_reduce(reduced_union)
                dist.all_reduce(reduced_target)

                intersection_meter.update(reduced_intersection.cpu().numpy())
                union_meter.update(reduced_union.cpu().numpy())
            else:
                intersection_meter.update(intersection)
                union_meter.update(union)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    return mIOU, iou_class


def main():
    parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument('--save-map', type=str, default=False)

    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg['save_map'] = args.save_map
    
    model = DeepLabV3Plus(cfg)
    ckpt = torch.load(args.ckpt_path)['model']
    ckpt = remove_module_prefix(ckpt) if cfg['dataset'] != 'pascal' else ckpt
    model.load_state_dict(ckpt)
    model.cuda()
    print('Total params: {:.1f}M\n'.format(count_params(model)))

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)
    
    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, save_path=args.ckpt_path)

    for (cls_idx, iou) in enumerate(iou_class):
        print('***** Evaluation ***** >>>> Class [{:} {:}] '
                    'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
    print('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
    
    
if __name__ == '__main__':
    main()

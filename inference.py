import os
import yaml
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.utils import count_params, color_map
from dataset.transform import normalize


class CustomDataset(Dataset):
    def __init__(self, img_folder, size=None):
        self.imgs = sorted(glob(img_folder + '/*'))
        self.size = size

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        name = os.path.basename(img_path)
        img = Image.open(img_path).convert('RGB')

        w, h = img.size
        
        if self.size is not None:
            # Determine the new dimensions while maintaining aspect ratio
            if max(w, h) < self.size:
                if h > w:
                    new_h = self.size
                    new_w = int(self.size * w / h)
                else:
                    new_w = self.size
                    new_h = int(self.size * h / w)
                img = img.resize((new_w, new_h), Image.BILINEAR)
        
        return normalize(img), name

    def __len__(self):
        return len(self.imgs)

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

def inference(model, loader, mode, cfg, save_path):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    palette = color_map(cfg['dataset'])
    
    if cfg['save_map']:
        save_path = save_path.split('.pth')[0]    # checkpoints/pascal_92_73.7.pth -> checkpoints/pascal_92_73.7
        os.makedirs(os.path.join(save_path, 'mask'), exist_ok=True)         # for mask
        os.makedirs(os.path.join(save_path, 'color_mask'), exist_ok=True)   # for colorized mask

    with torch.no_grad():
        for img, name in tqdm(loader):

            img = img.cuda()
            name = Path(name[0]).stem
            
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
                pred_map = pred[0].cpu().numpy().astype(np.uint8)
                pred_map = Image.fromarray(pred_map)
                pred_colormap = pred_map.convert('P')
                pred_colormap.putpalette(palette)
                
                pred_map.save(os.path.join(save_path, 'mask', name+'.png'))
                pred_colormap.save(os.path.join(save_path, 'color_mask', name+'.png'))


def main():
    parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
    parser.add_argument('--custom_dataset', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument('--save-map', type=str, default=True)
    parser.add_argument('--img-size', type=int, default=500)

    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg['save_map'] = args.save_map
    
    model = DeepLabV3Plus(cfg)
    ckpt = torch.load(args.ckpt_path)['model']
    ckpt = remove_module_prefix(ckpt) if cfg['dataset'] != 'pascal' else ckpt
    model.load_state_dict(ckpt)
    model.cuda()
    print('Total params: {:.1f}M\n'.format(count_params(model)))

    customset = CustomDataset(img_folder=args.custom_dataset, size=args.img_size)
    customloader = DataLoader(customset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)
    
    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    inference(model, customloader, eval_mode, cfg, save_path=args.ckpt_path)
    
if __name__ == '__main__':
    main()

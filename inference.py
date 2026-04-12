"""Inference and evaluation
"""
import os
import sys
import argparse
import numpy as np
import torch
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score

import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss




TRIMAP_COLORS = np.array([
    [255, 255, 255],   # 0 = foreground  (white)
    [0,   0,   0  ],   # 1 = background  (black)
    [128, 128, 128],   # 2 = uncertain   (grey)
], dtype=np.uint8)


def mask_to_rgb(mask_2d):
    return TRIMAP_COLORS[mask_2d.clip(0, 2)]


def load_checkpoint(model, path, device):
    ckpt  = torch.load(path, map_location=device)
    state = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def get_val_loader(data_root, batch_size=16):
    full = OxfordIIITPetDataset(root=data_root, split='trainval')
    val_size   = int(0.2 * len(full))
    train_size = len(full) - val_size
    _, val_data = random_split(
        full, [train_size, val_size],
        generator=torch.Generator().manual_seed(6)
    )
    loader = DataLoader(val_data, batch_size=batch_size,
                        shuffle=False, num_workers=2)
    return loader, full.class_to_idx


def get_inference_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def batch_dice(logits, gt, num_classes=3, eps=1e-6):
    B, C, H, W = logits.shape
    probs   = logits.softmax(dim=1)
    oh      = torch.zeros(B, C, H, W, device=logits.device)
    oh.scatter_(1, gt.unsqueeze(1), 1)
    probs   = probs.view(B, C, -1)
    oh      = oh.view(B, C, -1)
    inter   = (probs * oh).sum(2)
    denom   = probs.sum(2) + oh.sum(2)
    return ((2 * inter + eps) / (denom + eps)).mean().item()


def to_xyxy(box):
    cx, cy, w, h = box
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]




def run_section_27(mt_model, images_dir, class_names, device, wandb_project):
    transform = get_inference_transform()

    image_files = [
        f for f in sorted(os.listdir(images_dir))
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if len(image_files) == 0:
        print(f"[2.7] No images found in {images_dir}. "
              "Download pet images and place them there.")
        return

    print(f"[2.7] Found {len(image_files)} images: {image_files}")

    run = wandb.init(project=wandb_project, name='2.7-wild-pipeline', reinit=True)
    table = wandb.Table(columns=[
        'Filename', 'Original', 'Predicted_Breed', 'Confidence',
        'BBox_Image', 'Segmentation_Mask', 'Notes'
    ])

    for fname in image_files:
        fpath   = os.path.join(images_dir, fname)
        img_pil = Image.open(fpath).convert('RGB')
        img_np  = np.array(img_pil)
        img_224 = np.array(img_pil.resize((224, 224)))

        tensor = transform(image=img_np)['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            out = mt_model(tensor)

        # Classification
        probs   = out['classification'].softmax(dim=-1)[0].cpu()
        top_idx = probs.argmax().item()
        conf    = probs[top_idx].item()
        breed   = class_names[top_idx] if top_idx < len(class_names) else str(top_idx)

        # Top-3 predictions for notes
        top3_vals, top3_idx = probs.topk(3)
        top3_str = ' | '.join(
            f"{class_names[i.item()]}({v.item():.1%})"
            for v, i in zip(top3_vals, top3_idx)
        )

        # Localization 
        cx, cy, w, h = out['localization'][0].cpu().tolist()
        x1 = max(0, min(224, cx - w/2))
        y1 = max(0, min(224, cy - h/2))
        x2 = max(0, min(224, cx + w/2))
        y2 = max(0, min(224, cy + h/2))

        bbox_wb = wandb.Image(img_224, boxes={'predictions': {
            'box_data': [{
                'position': {'minX': x1/224, 'minY': y1/224, 'maxX': x2/224, 'maxY': y2/244},
                'class_id': 0,
                'box_caption': f'{breed} {conf:.0%}'
            }],
            'class_labels': {0: breed}
        }})

        # Segmentation 
        seg_pred = out['segmentation'].argmax(dim=1)[0].cpu().numpy()
        seg_rgb  = mask_to_rgb(seg_pred)

        # Simple analysis note
        box_area  = max(0, x2-x1) * max(0, y2-y1)
        image_area = 224 * 224
        coverage  = box_area / image_area
        if conf < 0.3:
            note = 'Low confidence — unusual pose or lighting'
        elif coverage < 0.05:
            note = 'Small bbox — possible scale issue'
        elif coverage > 0.9:
            note = 'Very large bbox — background confusion'
        else:
            note = f'OK — top3: {top3_str}'

        table.add_data(
            fname,
            wandb.Image(img_224, caption=fname),
            breed,
            round(conf, 4),
            bbox_wb,
            wandb.Image(seg_rgb, caption='Predicted trimap'),
            note
        )
        print(f"  {fname}: {breed} ({conf:.1%}) — bbox coverage {coverage:.1%}")

    wandb.log({'2.7_wild_pipeline': table})
    wandb.finish()
    print('[2.7] Done.')





def parse_args():
    parser = argparse.ArgumentParser(description='Inference — Sections 2.7 and 2.8')
    parser.add_argument('--images_dir',    type=str, default='./wild_images',
                        help='Folder containing downloaded pet images (2.7)')
    parser.add_argument('--data',          type=str, default='./data',
                        help='Oxford-IIIT Pet dataset root')
    parser.add_argument('--ckpt_dir',      type=str, default='./checkpoints',
                        help='Directory containing classifier/localizer/unet .pth files')
    parser.add_argument('--wandb_project', type=str, default='da6401')
    parser.add_argument('--batch_size',    type=int, default=16)
    parser.add_argument('--section',       type=str, default='2.7',
                        choices=['2.7'],
                        help='Which section to run')
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    clf_path = os.path.join(args.ckpt_dir, 'classifier.pth')
    loc_path = os.path.join(args.ckpt_dir, 'localizer.pth')
    unet_path = os.path.join(args.ckpt_dir, 'unet.pth')

    for p in [clf_path, loc_path, unet_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f'Checkpoint not found: {p}\n'
                f'Place classifier.pth / localizer.pth / unet.pth in {args.ckpt_dir}'
            )

    # Load all three models
    print('Loading checkpoints...')
    clf  = load_checkpoint(VGG11Classifier(dropout_p=0.5, batch_norm=True),
                           clf_path, device).to(device)
    loc  = load_checkpoint(VGG11Localizer(dropout_p=0.5, batch_norm=True),
                           loc_path, device).to(device)
    unet = load_checkpoint(VGG11UNet(num_classes=3, dropout_p=0.5, batch_norm=True),
                           unet_path, device).to(device)

    # MultiTask model (for 2.7)
    mt = MultiTaskPerceptionModel(
        classifier_path=clf_path,
        localizer_path=loc_path,
        unet_path=unet_path,
        batch_norm=True,
        dropout_p=0.5,
    ).to(device)
    mt.eval()

    
    val_loader, class_to_idx = get_val_loader(args.data, args.batch_size)
    class_names = sorted(class_to_idx.keys())
    iou_fn = IoULoss(reduction='none')

    os.makedirs(args.images_dir, exist_ok=True)

    if args.section in ('2.7'):
        print('\n=== Section 2.7 — Wild Pipeline ===')
        run_section_27(mt, args.images_dir, class_names, device, args.wandb_project)


if __name__ == '__main__':
    main()
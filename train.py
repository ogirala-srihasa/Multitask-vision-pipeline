"""Training entrypoint
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import wandb
import numpy as np
from sklearn.metrics import f1_score
from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss
import argparse

def dice_loss(pred_logits, target, num_classes=3, eps=1e-6):
    """
    pred_logits: [B, num_classes, H, W] raw logits
    target:      [B, H, W] integer class indices {0,1,2}
    returns:     scalar loss value in [0, 1]
    """
    B, C, H, W = pred_logits.shape
    
    # converting logits to probabilities
    pred_prob = torch.softmax(pred_logits,dim = 1)
    
    # one-hot encoding target to [B, C, H, W]
    one_hot = torch.zeros(B, num_classes, H, W, device=target.device)
    one_hot.scatter_(1, target.unsqueeze(1), 1)
    
    # computing per-class Dice score
    # flatten spatial dims for easier computation
    probs = pred_prob.view(B, C, -1)
    one_hot = one_hot.view(B, C, -1)

    #computing intersection and sums per class
    intersection = torch.sum(probs * one_hot, dim=2)
    pred_sum = torch.sum(probs, dim=2)
    target_sum = torch.sum(one_hot, dim=2)

    dice_score = (2. * intersection + eps) / (pred_sum + target_sum + eps)
    
    # Dice loss = 1 - mean(Dice scores across classes and batch)
    return 1 - dice_score.mean()

def dice_score(pred_logits, target, num_classes=3, eps=1e-6):
    """
    pred_logits: [B, num_classes, H, W] raw logits
    target:      [B, H, W] integer class indices {0,1,2}
    returns:     scalar score value
    """
    B, C, H, W = pred_logits.shape
    
    # converting logits to probabilities
    pred_prob = torch.softmax(pred_logits,dim = 1)
    
    # one-hot encoding target to [B, C, H, W]
    one_hot = torch.zeros(B, num_classes, H, W, device=target.device)
    one_hot.scatter_(1, target.unsqueeze(1), 1)
    
    # computing per-class Dice score
    # flatten spatial dims for easier computation
    probs = pred_prob.view(B, C, -1)
    one_hot = one_hot.view(B, C, -1)

    #computing intersection and sums per class
    intersection = torch.sum(probs * one_hot, dim=2)
    pred_sum = torch.sum(probs, dim=2)
    target_sum = torch.sum(one_hot, dim=2)

    dice_score = (2. * intersection + eps) / (pred_sum + target_sum + eps)
    
    return dice_score.mean()

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('-t','--task',type = str, default='classification', choices=['classification','localization','segmentation'])
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type = int ,default= 128, help='Mini-batch size')
    parser.add_argument('-lr', '--learning_rate', type= float, default= 0.001, help= 'Learning rate for optimizer')
    parser.add_argument('-d','--data', type = str, default= "./data")
    parser.add_argument('-bn', '--batch_norm', type = str ,default= 'true',choices=['true','false'] ,help='   Batch Normalization ')
    parser.add_argument('-p', '--dropout', type= float, default= 0.5, help= 'dropout rate')
    parser.add_argument('-wp', '--wandb_project', type=str, default='da6401')


    
    return parser.parse_args()

def train_one_epoch_classification(model, loader, criterion, optimizer, device):
    model.train()                          
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels, bboxes, masks in loader:
        # move data to device
        images = images.to(device)
        labels = labels.to(device)

        # 5-step inner loop
        optimizer.zero_grad()
        logits = model(images)             # [B, 37]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # accumulate metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, f1

def validate_classification(model, loader, criterion, device):
    model.eval()                          
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels, bboxes, masks in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, f1

def train_one_epoch_localization(model, loader, L1_loss, mean_iou_loss, sample_iou_loss, optimizer, device):
    model.train()
    total_loss = 0.0
    total_iou = 0.0

    for images, labels, bboxes, masks in loader:
        images = images.to(device)
        bboxes = bboxes.to(device)      # target is bboxes now, not labels

        optimizer.zero_grad()
        pred_boxes = model(images)       # [B, 4]

        # combined loss
        loss_l1 = L1_loss(pred_boxes, bboxes)
        loss_iou = mean_iou_loss(pred_boxes, bboxes)
        loss = loss_l1 + loss_iou

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            per_sample_iou = 1.0 - sample_iou_loss(pred_boxes.detach(), bboxes) 
            total_iou += per_sample_iou.mean().item() 

    avg_loss = total_loss / len(loader)
    avg_iou  = total_iou / len(loader)
    return avg_loss, avg_iou

def validate_localization(model, loader,L1_loss,mean_iou_loss,sample_iou_loss,device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0 
    with torch.no_grad():
        for images, labels, bboxes, masks in loader:
            images = images.to(device)
            bboxes = bboxes.to(device)      # target is bboxes now, not labels

            pred_boxes = model(images)       # [B, 4]

            # combined loss
            loss_l1 = L1_loss(pred_boxes, bboxes)
            loss_iou = mean_iou_loss(pred_boxes, bboxes)
            loss = loss_l1 + loss_iou

            total_loss += loss.item()
            per_sample_iou = 1.0 - sample_iou_loss(pred_boxes.detach(), bboxes) 
            total_iou += per_sample_iou.mean().item() 

    avg_loss = total_loss / len(loader)
    avg_iou  = total_iou / len(loader)
    return avg_loss, avg_iou

def train_one_epoch_segmentation(model, loader, ce_loss, optimizer, device):
    model.train()
    total_loss = 0.0
    total_dice_score = 0.0

    for images, labels, bboxes, masks in loader:
        images = images.to(device)
        masks = masks.to(device)      

        optimizer.zero_grad()
        pred_mask = model(images)      

        # combined loss
        loss_ce = ce_loss(pred_mask, masks)
        loss_dice= dice_loss(pred_mask, masks)
        loss = loss_ce + loss_dice

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            total_dice_score += dice_score(pred_mask,masks).item()

    avg_loss = total_loss / len(loader)
    avg_dice_score  = total_dice_score/ len(loader)
    return avg_loss, avg_dice_score

def validate_segmentation(model, loader,ce_loss,device):
    model.eval()
    total_loss = 0.0
    total_dice_score = 0.0 
    with torch.no_grad():
        for images, labels, bboxes, masks in loader:
            images = images.to(device)
            masks = masks.to(device)      

            pred_mask = model(images)       

            # combined loss
            loss_mse = ce_loss(pred_mask, masks)
            loss_iou = dice_loss(pred_mask, masks)
            loss = loss_mse + loss_iou

            total_loss += loss.item()
            total_dice_score += dice_score(pred_mask,masks).item()

    avg_loss = total_loss / len(loader)
    avg_dice_score  = total_dice_score / len(loader)
    return avg_loss, avg_dice_score

def main():
    #collecting CLI args
    args = parse_arguments()
    #initiating wandb
    wandb.init(project=args.wandb_project, config=args)
    #connecting GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #finding bactcnorm

    if args.batch_norm == 'true': 
        batch_norm = True
    else:
        batch_norm = False

    # loading data
    full_train = OxfordIIITPetDataset(args.data, 'trainval')

    val_size   = int(0.2 * len(full_train))
    train_size = len(full_train) - val_size

    train_data, val_data = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(6)
    )

    print(f"Train: {len(train_data)}, Val: {len(val_data)}") 
    optimal_workers = min(4, os.cpu_count())
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True, # Shuffle batches every epoch
        num_workers=optimal_workers,
        pin_memory= True
    )
    
    val_loader = DataLoader(
        val_data, 
        batch_size=args.batch_size, 
        shuffle=False, # No need to shuffle validation
        num_workers=optimal_workers,
        pin_memory= True
    )

    #choosing model
    if(args.task == 'classification'):
        model = VGG11Classifier(dropout_p= args.dropout, batch_norm= batch_norm)
        loss = nn.CrossEntropyLoss()
        model.to(device=device)
        optimizer = optim.Adam(model.parameters(), lr= args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, factor=0.5
        )
        best_val_f1 = 0
        for epoch in range(args.epochs):
            train_loss, train_f1 = train_one_epoch_classification(model, train_loader, loss, optimizer, device)
            val_loss, val_f1     = validate_classification(model, val_loader, loss, device)
            scheduler.step(val_f1)

            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_f1={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_val_f1,
                }, "checkpoints/classifier.pth")
                print(f"Saved best model (val_f1={best_val_f1:.4f})")

            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/f1": train_f1,
                "val/loss": val_loss,
                "val/f1": val_f1,
                "best_val/f1": best_val_f1
            })
    elif args.task == "localization":
        model = VGG11Localizer(dropout_p=args.dropout, batch_norm=batch_norm)
        iouloss_none = IoULoss(reduction= 'none')
        iouloss_mean = IoULoss(reduction='mean')
        L1loss = nn.L1Loss()

        # transfer learning
        if os.path.exists("checkpoints/classifier.pth"):
            checkpoint = torch.load("checkpoints/classifier.pth", map_location=device)
            full_state = checkpoint.get("state_dict", checkpoint)
            encoder_state = {
                k.replace("VGGhead.", ""): v
                for k, v in full_state.items()
                if k.startswith("VGGhead.")
            }
            model.VGGhead.load_state_dict(encoder_state)
            print("Loaded encoder weights from classifier.pth ")
            model.to(device=device)
            for param in model.VGGhead.parameters():
                param.requires_grad = False
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = optim.Adam(trainable_params, lr=args.learning_rate)         
            # optimizer = optim.Adam([
            #     {"params": model.VGGhead.parameters(), "lr": args.learning_rate * 0.1},
            #     {"params": model.layer1.parameters(), "lr": args.learning_rate},
            #     {"params": model.layer2.parameters(), "lr": args.learning_rate},
            #     {"params": model.layer3.parameters(), "lr": args.learning_rate},
            # ])
        else:
            print("No classifier checkpoint found, training from scratch")
            model.to(device=device)
            optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
        best_val_iou = 0
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, factor=0.5
        )
        for epoch in range(args.epochs):
            train_loss, train_iou = train_one_epoch_localization(model, train_loader,L1_loss= L1loss, mean_iou_loss=iouloss_mean,sample_iou_loss=iouloss_none, optimizer= optimizer,device= device)
            val_loss, val_iou     = validate_localization(model, val_loader, L1_loss = L1loss,device= device, mean_iou_loss = iouloss_mean, sample_iou_loss= iouloss_none)
            scheduler.step(val_iou)
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_iou={val_iou:.4f}")

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save({
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_val_iou,
                }, "checkpoints/localizer.pth")
                print(f"Saved best model (val_iou={best_val_iou:.4f})")

            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/iou": train_iou,
                "val/loss": val_loss,
                "val/iou": val_iou,
                "best_val/iou": best_val_iou
            })
    else:
        model = VGG11UNet(dropout_p=args.dropout, batch_norm=batch_norm)
        ce_loss = nn.CrossEntropyLoss()

        # transfer learning
        if os.path.exists("checkpoints/classifier.pth"):
            checkpoint = torch.load("checkpoints/classifier.pth", map_location=device)
            full_state = checkpoint.get("state_dict", checkpoint)
            encoder_state = {
                k.replace("VGGhead.", ""): v
                for k, v in full_state.items()
                if k.startswith("VGGhead.")
            }
            model.VGGhead.load_state_dict(encoder_state)
            print("Loaded encoder weights from classifier.pth ")
            model.to(device=device)
            decoder_params = list(model.up1.parameters()) + list(model.dec1.parameters()) + \
                 list(model.up2.parameters()) + list(model.dec2.parameters()) + \
                 list(model.up3.parameters()) + list(model.dec3.parameters()) + \
                 list(model.up4.parameters()) + list(model.dec4.parameters()) + \
                 list(model.up5.parameters()) + list(model.dec5.parameters()) + \
                 list(model.outConv.parameters()
            )

            # for param in model.VGGhead.parameters():
            #     param.requires_grad = False
            # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            # optimizer = optim.Adam(trainable_params, lr=args.learning_rate) 
            optimizer = optim.Adam([
                {"params": model.VGGhead.parameters(), "lr": args.learning_rate * 0.1},
                {"params": decoder_params, "lr": args.learning_rate},
            ])
        else:
            print("No classifier checkpoint found, training from scratch")
            model.to(device=device)
            optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
        best_val_dice_score = 0
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, factor=0.5
        )
        for epoch in range(args.epochs):
            train_loss, train_dice_score = train_one_epoch_segmentation(model, train_loader,ce_loss= ce_loss, optimizer= optimizer,device= device)
            val_loss, val_dice_score    = validate_segmentation(model, val_loader,ce_loss=ce_loss,device= device)
            scheduler.step(val_dice_score)
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_dice_score={val_dice_score:.4f}")

            if val_dice_score > best_val_dice_score:
                best_val_dice_score = val_dice_score
                torch.save({
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_val_dice_score,
                }, "checkpoints/unet.pth")
                print(f"Saved best model (val_dice_score={best_val_dice_score:.4f})")

            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/dice_score": train_dice_score,
                "val/loss": val_loss,
                "val/dice_score": val_dice_score,
                "best_val/dice_score": best_val_dice_score
            })




    wandb.finish()


if __name__ == "__main__":
    main()

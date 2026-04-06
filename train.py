"""Training entrypoint
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import numpy as np
from sklearn.metrics import f1_score
from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
import argparse

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('-t','--task',type = str, default='classification', choices=['classification'])
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type = int ,default= 128, help='Mini-batch size')
    parser.add_argument('-lr', '--learning_rate', type= float, default= 0.001, help= 'Learning rate for optimizer')
    parser.add_argument('-d','--data', type = str, default= "./data")
    parser.add_argument('-bn', '--batch_norm', type = bool ,default= True, help='   Batch Normalization ')
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


def main():
    #collecting CLI args
    args = parse_arguments()
    #initiating wandb
    wandb.init(project=args.wandb_project, config=args)
    #connecting GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # loading data
    train_data = OxfordIIITPetDataset(args.data, 'trainval')
    val_data = OxfordIIITPetDataset(args.data, 'test')
    optimal_workers = os.cpu_count()
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
        model = VGG11Classifier(dropout_p= args.dropout, batch_norm= args.batch_norm)
        loss = nn.CrossEntropyLoss()
        model.to(device=device)
        optimizer = optim.Adam(model.parameters(), lr= args.learning_rate)
        best_val_f1 = 0
        for epoch in range(args.epochs):
            train_loss, train_f1 = train_one_epoch_classification(model, train_loader, loss, optimizer, device)
            val_loss, val_f1     = validate_classification(model, val_loader, loss, device)

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
    

    wandb.finish()


if __name__ == "__main__":
    main()






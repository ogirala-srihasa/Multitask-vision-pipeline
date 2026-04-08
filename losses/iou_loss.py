"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Invalid reduction: {reduction}")

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        # TODO: implement IoU loss.

        # Unpacking Pred and target boxes
        pcx = pred_boxes[:, 0]    
        pcy = pred_boxes[:, 1]    
        pw  = pred_boxes[:, 2]    
        ph  = pred_boxes[:, 3] 

        tcx = target_boxes[:, 0]    
        tcy = target_boxes[:, 1]    
        tw  = target_boxes[:, 2]    
        th  = target_boxes[:, 3] 

        #converting them to min-max format

        px1 = pcx - pw/2  
        py1 = pcy - ph/2        
        px2 = pcx + pw/2        
        py2 = pcy + ph/2

        tx1 = tcx - tw/2        
        ty1 = tcy - th/2        
        tx2 = tcx + tw/2        
        ty2 = tcy + th/2

        #finding the intersection

        inter_x1 = torch.max(px1, tx1)    
        inter_y1 = torch.max(py1, ty1)    
        inter_x2 = torch.min(px2, tx2)    
        inter_y2 = torch.min(py2, ty2)   

        #finding intersection weight and height

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)

        #Calculating intersection area
        intersection = inter_w * inter_h

        #Calculating Union area
        area_pred   = pw * ph
        area_target = tw * th
        union = area_pred + area_target - intersection

        IoU = intersection / (union + self.eps) 
        loss = 1.0 - IoU        # range: [0, 1]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss







        
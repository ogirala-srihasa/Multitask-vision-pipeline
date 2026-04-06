"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import xml.etree.ElementTree as ET
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import numpy as np
import torch

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    def __init__(self, root: str, split: str):
        super().__init__()
        self.root = root
        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "annotations", "trimaps")
        self.xmls_dir = os.path.join(root, "annotations", "xmls")
        self.annotations_dir = os.path.join(root, "annotations")
        self.files= []
        self.classes = set()
        self.class_to_idx = {}
        self.transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        # Building class_to_idx mapping
        with open(os.path.join(self.annotations_dir, "list.txt"), "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue                      # skip comments and blank lines

                parts = line.split()              # split on whitespace
                image_name = parts[0]            # e.g. "Abyssinian_1"
                class_id   = int(parts[1]) - 1   # convert to 0-indexed (1→0, 2→1 ...)
                clas = "_".join(image_name.split("_")[:-1])
                self.class_to_idx[clas] = class_id
                self.classes.add(clas)

        #collecting the files
        if(split == 'trainval'):
            with open(os.path.join(self.annotations_dir, "trainval.txt"), "r") as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue
                    filename = line.split()[0]
                    if os.path.exists(os.path.join(self.xmls_dir, f"{filename}.xml")) and os.path.exists(os.path.join(self.masks_dir, f"{filename}.png")):
                        self.files.append(filename)
        elif split == 'test':
            with open(os.path.join(self.annotations_dir, "test.txt"), "r") as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue
                    filename = line.split()[0]
                    if os.path.exists(os.path.join(self.xmls_dir, f"{filename}.xml")) and os.path.exists(os.path.join(self.masks_dir, f"{filename}.png")):
                        self.files.append(filename)
        else:
            raise ValueError(f"Invalid split: {split}. Choose 'trainval' or 'test'.")
        self.files.sort()

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = self.files[idx]

        # Loading image
        img_path = os.path.join(self.images_dir, f"{filename}.jpg")
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        #Loading Mask
        mask_path = os.path.join(self.masks_dir, f"{filename}.png")
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)

        # Loading XML for bbox
        xml_path = os.path.join(self.xmls_dir, f"{filename}.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        xmin = int(root.find('.//xmin').text)
        xmax = int(root.find('.//xmax').text)
        ymin = int(root.find('.//ymin').text)
        ymax = int(root.find('.//ymax').text)

        #Creating a min-max Bounding Box
        bbox = [xmin, ymin, xmax, ymax]

        breed_name = "_".join(filename.split("_")[:-1])
        label_idx = self.class_to_idx[breed_name]

        #applying transformation
        transformed = self.transform(
            image = img,
            mask = mask,
            bboxes = [bbox],
            class_labels = [label_idx]
        )
        img = transformed['image']
        mask = transformed['mask']
        bbox_valid = True
        if(len(transformed['bboxes']) > 0):
            bbox = transformed['bboxes'][0]
        else:
            bbox = [0, 0, 0, 0]
            bbox_valid = False

        if bbox_valid: # Only convert if the box wasn't cropped out
            bbox = [
                (bbox[0] + bbox[2]) / 2.0,  # x_center
                (bbox[1] + bbox[3]) / 2.0,  # y_center
                (bbox[2] - bbox[0]),        # width
                (bbox[3] - bbox[1])         # height
            ]
        mask = torch.clamp(torch.as_tensor(mask, dtype=torch.long) - 1, min=0, max=2)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        label_idx = torch.tensor(label_idx, dtype=torch.long)
        return img, label_idx, bbox_tensor, mask
    
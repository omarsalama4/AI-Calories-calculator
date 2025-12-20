import os
import re
import random
from tqdm import tqdm

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

# ==================== REPRODUCIBILITY ====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# ==================== DATA AUGMENTATION ====================
class DataAugmentation:
    def __init__(self, image_size=256, is_train=True):
        self.image_size = image_size
        self.is_train = is_train
    
    def __call__(self, image, mask):
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        if self.is_train:
            # Horizontal flip
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            
            # Vertical flip
            if random.random() > 0.5:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)
            
            # Rotation
            if random.random() > 0.5:
                angle = random.uniform(-45, 45)
                h, w = image.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h))
                mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
            
            # Brightness/contrast
            if random.random() > 0.7:
                alpha = random.uniform(0.8, 1.2)
                beta = random.uniform(-20, 20)
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            # Gaussian noise
            if random.random() > 0.8:
                noise = np.random.normal(0, 15, image.shape).astype(np.int16)
                image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # To tensors
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        return image, mask


# ==================== DATASET CLASS ====================
def extract_id(filename):
    """
    Extract numeric ID from filenames like:
    1.jpg, 23.png, 7_mask.png
    """
    match = re.search(r'\d+', filename)
    return match.group() if match else None

class FruitSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_size=256, is_train=True):
        self.root_dir = root_dir
        self.transform = DataAugmentation(image_size, is_train)
        self.samples = []
        
        # Validate path
        if not os.path.exists(root_dir):
            raise ValueError(f"Path does not exist: {root_dir}")
        
        print(f"Searching in: {root_dir}")
        
        fruit_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        print(f"Found {len(fruit_folders)} fruit folders: {fruit_folders}")
        
        total_matched = 0
        for fruit_name in fruit_folders:
            fruit_path = os.path.join(root_dir, fruit_name)
            images_dir = os.path.join(fruit_path, 'Images')
            masks_dir = os.path.join(fruit_path, 'Mask')
            
            if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
                print(f"Skipping {fruit_name}: missing Images or Mask folder")
                continue
            
            image_map = {}
            for f in os.listdir(images_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_id = extract_id(f)
                    if img_id is not None:
                        image_map[img_id] = os.path.join(images_dir, f)

            mask_map = {}
            for f in os.listdir(masks_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    mask_id = extract_id(f)
                    if mask_id is not None:
                        mask_map[mask_id] = os.path.join(masks_dir, f)

            matched = 0
            for img_id, img_path in image_map.items():
                if img_id in mask_map:
                    self.samples.append({
                        'image': img_path,
                        'mask': mask_map[img_id],
                        'fruit': fruit_name
                    })
                    matched += 1
            total_matched += matched
            print(f"  {fruit_name}: {matched} pairs")

        if len(self.samples) == 0:
            raise ValueError(f"No valid image-mask pairs found in {root_dir}. Check your folder structure!")
            
        print(f"Total loaded: {len(self.samples)} pairs (total matched across folders: {total_matched})\n")
        self.calculate_class_weights()
        
    def calculate_class_weights(self, sample_size=100):
        total_fg = 0
        total_bg = 0
        
        indices = random.sample(range(len(self.samples)), min(sample_size, len(self.samples)))
        
        for idx in indices:
            mask = cv2.imread(self.samples[idx]['mask'], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            fg_pixels = np.sum(mask > 127)
            bg_pixels = np.sum(mask <= 127)
            total_fg += fg_pixels
            total_bg += bg_pixels
        
        # avoid division by zero
        self.pos_weight = float(total_bg / (total_fg + 1e-6))
        print(f"Class imbalance (bg/fg) estimate (pos_weight): {self.pos_weight:.2f}\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = cv2.imread(sample['image'])
        if image is None:
            raise RuntimeError(f"Failed to read image: {sample['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {sample['mask']}")
        mask = (mask > 127).astype(np.float32)
        
        image, mask = self.transform(image, mask)
        
        return image, mask


# ==================== LOSSES ====================
class DiceLoss(nn.Module):
    """
    Per-sample Dice loss. Accepts logits (from_logits=True) or probabilities.
    Returns mean loss over batch.
    """
    def __init__(self, smooth=1e-6, from_logits=True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, preds, targets):
        # preds: (B,1,H,W) logits or probs
        if self.from_logits:
            probs = torch.sigmoid(preds)
        else:
            probs = preds

        B = probs.shape[0]
        probs = probs.view(B, -1)
        targets = targets.view(B, -1).float()

        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2. * intersection + self.smooth) / (denom + self.smooth)
        loss = 1.0 - dice
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    BCEWithLogits + Dice. Accepts pos_weight as float or tensor.
    """
    def __init__(self, pos_weight=None, dice_from_logits=True, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.pos_weight = pos_weight  # float or tensor
        self.dice = DiceLoss(from_logits=dice_from_logits)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        # ensure pos_weight tensor on same device
        if self.pos_weight is not None and not isinstance(self.pos_weight, torch.Tensor):
            pw = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        elif isinstance(self.pos_weight, torch.Tensor):
            pw = self.pos_weight.to(logits.device)
        else:
            pw = None

        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        bce = bce_loss_fn(logits, target)
        dice_loss = self.dice(logits, target)  # dice expects logits if initialized that way
        return self.bce_weight * bce + self.dice_weight * dice_loss


# ==================== METRICS ====================
def soft_dice(pred, target, smooth=1e-6):
    """
    Soft Dice (no threshold) — use during TRAINING monitoring.
    Accepts logits.
    Returns mean Dice coefficient over batch (not loss).
    """
    probs = torch.sigmoid(pred)
    B = probs.shape[0]
    probs = probs.view(B, -1)
    target = target.view(B, -1).float()

    intersection = (probs * target).sum(dim=1)
    denom = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2. * intersection + smooth) / (denom + smooth)
    return dice.mean()


def hard_dice(pred, target, threshold=0.5, smooth=1e-6):
    """
    Hard Dice (thresholded) — use during VALIDATION final metrics.
    Accepts logits.
    Returns mean Dice coefficient over batch (not loss).
    """
    probs = torch.sigmoid(pred)
    B = probs.shape[0]
    preds_bin = (probs > threshold).float().view(B, -1)
    target_bin = (target > 0.5).float().view(B, -1)

    intersection = (preds_bin * target_bin).sum(dim=1)
    denom = preds_bin.sum(dim=1) + target_bin.sum(dim=1)
    dice = (2. * intersection + smooth) / (denom + smooth)
    return dice.mean()


def iou_score(preds, targets, threshold=None, smooth=1e-6):
    """
    If threshold is None -> soft IoU using probabilities.
    If threshold given -> hard IoU with thresholding.
    Returns mean IoU over batch.
    """
    probs = torch.sigmoid(preds)
    B = probs.shape[0]
    probs = probs.view(B, -1)
    targets = targets.view(B, -1).float()

    if threshold is not None:
        preds_bin = (probs > threshold).float()
        inter = (preds_bin * targets).sum(dim=1)
        union = preds_bin.sum(dim=1) + targets.sum(dim=1) - inter
    else:
        inter = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1) - inter

    iou = (inter + smooth) / (union + smooth)
    return iou.mean()


# ==================== EARLY STOPPING ====================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


# ==================== TRAIN / VALID LOOPS ====================
def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0
    
    for images, masks in tqdm(loader, desc="Train", leave=False):
        images, masks = images.to(device), masks.to(device)
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_dice += soft_dice(outputs, masks).item()
        total_iou += iou_score(outputs, masks).item()
        num_batches += 1
    
    if num_batches == 0:
        return 0.0, 0.0, 0.0
    return total_loss / num_batches, total_dice / num_batches, total_iou / num_batches


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Val", leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            total_dice += hard_dice(outputs, masks).item()
            total_iou += iou_score(outputs, masks, threshold=0.5).item()
            num_batches += 1
    
    if num_batches == 0:
        return 0.0, 0.0, 0.0
    return total_loss / num_batches, total_dice / num_batches, total_iou / num_batches


# ==================== MAIN ====================
def main():
    BATCH_SIZE = 8
    LR = 1e-4
    EPOCHS = 2
    IMG_SIZE = 256
    PATIENCE = 15
    NUM_WORKERS = 2
    ENCODER = 'resnet34'  # choose encoder carefully; resnet34 is a safe default

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}\n")
        except Exception:
            pass
    
    # Paths - change if necessary
    train_root = 'Data/Fruit/Train'
    val_root = 'Data/Fruit/Validation'
    
    train_data = FruitSegmentationDataset(train_root, IMG_SIZE, is_train=True)
    val_data = FruitSegmentationDataset(val_root, IMG_SIZE, is_train=False)
    
    pin_memory = True if device.type == 'cuda' else False
    train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=pin_memory)
    val_loader = DataLoader(val_data, BATCH_SIZE, shuffle=False, 
                           num_workers=NUM_WORKERS, pin_memory=pin_memory)
    
    # Model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None  # logits
    ).to(device)
    
    # pos_weight must be a tensor on same device
    pos_weight_tensor = torch.tensor(train_data.pos_weight, device=device, dtype=torch.float32)
    criterion = CombinedLoss(pos_weight=pos_weight_tensor, dice_from_logits=True)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    early_stop = EarlyStopping(PATIENCE)
    
    # Train
    best_dice = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': [], 'train_iou': [], 'val_iou': []}
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{EPOCHS}\n{'='*60}")
        
        tr_loss, tr_dice, tr_iou = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(tr_dice)
        history['val_dice'].append(val_dice)
        history['train_iou'].append(tr_iou)
        history['val_iou'].append(val_iou)
        
        print(f"\nTrain - Loss: {tr_loss:.4f}, Soft Dice: {tr_dice:.4f}, Soft IoU: {tr_iou:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Hard Dice: {val_dice:.4f}, Hard IoU: {val_iou:.4f}")
        
        # Save best
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': val_dice,
            'val_iou': val_iou,
            'encoder': ENCODER
        }

        if val_dice > best_dice + 1e-4:
            best_dice = val_dice
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "best_fruit_segmentation.pth"))
            print(f" Saved BEST model (Dice={best_dice:.4f})")

                
        if early_stop(val_loss):
            print(f"\n Early stop at epoch {epoch+1}")
            break


    
    print(f"\nDone! Best Val Dice: {best_dice:.4f}")
    
    import json
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=4)


if __name__ == '__main__':
    main()

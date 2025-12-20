import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import segmentation_models_pytorch as smp

# ==================== CONFIGURATION ====================
# Update these paths to match your project folders
MODEL_PATH = 'Fruit binary Segmentation\\best_fruit_segmentation.pth'  # Path to your saved .pth file
INPUT_FOLDER = 'Test Cases Structure/Integerated Test'                 # Folder containing images to test
OUTPUT_FOLDER = 'test_results'               # Folder where results will be saved
IMAGE_SIZE = 256                             # Must match training size
THRESHOLD = 0.5                              # Probability threshold for binary mask

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== MODEL CLASS ====================
class ModernUNet(nn.Module):
    """
    Same class definition as training to ensure weights load correctly.
    """
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', dropout=0.3):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )
        self.dropout = nn.Dropout2d(p=dropout)
    
    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return torch.sigmoid(x)

# ==================== UTILS ====================
def load_checkpoint(model, filename):
    print(f"Loading checkpoint from {filename}")
    try:
        checkpoint = torch.load(filename, map_location=DEVICE)
    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        exit()
    
    # 1. Extract state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Found checkpoint info: Epoch {checkpoint.get('epoch', '?')}, Val Dice: {checkpoint.get('val_dice', '?'):.4f}")
    else:
        state_dict = checkpoint

    # 2. Fix Key Mismatch
    # If the file has keys like "encoder..." but model expects "model.encoder...", fix it.
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_state_dict[key] = value
        elif not key.startswith("model.") and hasattr(model, 'model'):
            new_key = "model." + key
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    # 3. Load
    try:
        model.load_state_dict(new_state_dict)
        print("Success: Model weights loaded correctly!")
    except RuntimeError as e:
        print("Warning: Standard load failed, trying inner model load...")
        try:
            model.model.load_state_dict(state_dict)
            print("Success: Inner model weights loaded!")
        except Exception as e2:
            print(f"Error loading weights: {e}")
            exit()

    model.eval()
    return model

def visualize(image, mask, filename, save_dir):
    """
    Saves a side-by-side comparison: Original Image | Predicted Mask | Overlay
    Args:
        image: Numpy array (H, W, 3) in BGR format (OpenCV standard).
        mask: Numpy array (H, W). Binary mask (0 or 1).
    """
    # 1. Ensure image is uint8 (0-255)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # 2. Create Red Overlay
    # In OpenCV BGR, channels are [Blue, Green, Red]. We set Red (index 2) to 255.
    mask_colored = np.zeros_like(image)
    mask_colored[:, :, 2] = (mask * 255).astype(np.uint8) 
    
    # 3. Create Overlay (Blend Image + Mask)
    overlay = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
    
    # 4. Prepare Mask for stacking (Convert 1-channel to 3-channel grayscale)
    mask_3ch = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # 5. Stack: Original | Mask | Overlay
    combined = np.hstack((image, mask_3ch, overlay))
    
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, combined)

# ==================== MAIN INFERENCE LOOP ====================
def run_test():
    # 1. Setup Folders
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # 2. Load Model
    print("Initializing model...")
    # encoder_weights=None because we are loading our own trained weights
    model = ModernUNet(encoder_name='resnet34', encoder_weights=None) 
    model = model.to(DEVICE)
    
    model = load_checkpoint(model, MODEL_PATH)

    # 3. Define Preprocessing
    test_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # 4. Process Images
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' does not exist.")
        return

    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images in {INPUT_FOLDER}")

    for img_name in tqdm(image_files):
        img_path = os.path.join(INPUT_FOLDER, img_name)
        
        # Read Image (OpenCV reads in BGR format)
        original_img_bgr = cv2.imread(img_path)
        if original_img_bgr is None:
            print(f"Could not read {img_name}")
            continue
        
        # Convert to RGB for the Model (Model expects RGB)
        original_img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        augmented = test_transform(image=original_img_rgb)
        img_tensor = augmented['image'].unsqueeze(0).to(DEVICE) # Add batch dim
        
        # Inference
        with torch.no_grad():
            output = model(img_tensor)
            pred_mask = (output > THRESHOLD).float().cpu().numpy()
            
        # Remove batch/channel dims -> (H, W)
        pred_mask = pred_mask[0, 0, :, :]
        
        # Resize original BGR image to match model size for visualization
        # We use BGR here so colors look correct in the final saved image
        viz_img = cv2.resize(original_img_bgr, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Save Result
        visualize(viz_img, pred_mask, img_name, OUTPUT_FOLDER)

    print(f"\nProcessing complete! Check '{OUTPUT_FOLDER}' for results.")

if __name__ == '__main__':
    run_test()
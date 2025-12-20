import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

MODEL_PATH = 'checkpoints/best_fruit_segmentation_98.73.pth'
IMAGE_DIR = 'test data'
OUTPUT_DIR = 'predictions'
IMG_SIZE = 256
ENCODER = 'resnet34'   # must match training
THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# load model
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

model = smp.Unet(
    encoder_name=checkpoint.get('encoder', ENCODER),
    encoder_weights=None,   # no imagenet during inference
    in_channels=3,
    classes=1,
    activation=None        
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# preprocess 
def preprocess_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std

    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    return image

# Inference
with torch.no_grad():
    for fname in os.listdir(IMAGE_DIR):
        
        img_path = os.path.join(IMAGE_DIR, fname)
        image = preprocess_image(img_path).to(device)

        logits = model(image)
        probs = torch.sigmoid(logits)
        mask = (probs > THRESHOLD).float()

        mask_np = mask.squeeze().cpu().numpy() * 255
        mask_np = mask_np.astype(np.uint8)

        out_path = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(out_path, mask_np)

print('Inference complete.')

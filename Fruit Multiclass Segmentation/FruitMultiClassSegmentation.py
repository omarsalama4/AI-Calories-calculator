import cv2
import numpy as np
import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_FOLDER = "Test Cases Structure/Integerated Test"
OUTPUT_FOLDER = "test_results"   
MODEL_PATH = "Fruit Multiclass Segmentation\\best_model_multiclass.pth"    
IMG_HEIGHT = 512
IMG_WIDTH = 512
N_CLASSES = 31

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. CLASS MAPPING (Added this back)
# ==========================================
fruit_to_class = {
    "background": 0, "apple_Gala":1, "apple_Golden Delicious":2, "Avocado":3, "Banana":4,
    "Berry":5, "Burmese Grape":6, "Carambola":7, "Date Palm":8, "Dragon":9, "Elephant Apple":10,
    "Grape":11, "Green Coconut":12, "Guava":13, "Hog Plum":14, "Kiwi":15, "Lichi":16, "Malta":17,
    "Mango Golden Queen":18, "Mango_Alphonso":19, "Mango_Amrapali":20, "Mango_Bari":21,
    "Mango_Himsagar":22, "Olive":23, "Orange":24, "Palm":25, "Persimmon":26, "Pineapple":27,
    "Pomegranate":28, "Watermelon":29, "White Pear":30
}
# Create the reverse map: ID -> Name (e.g., 4 -> "Banana")
class_to_fruit = {v: k for k, v in fruit_to_class.items()}

# ==========================================
# 3. COLOR MAP SETUP
# ==========================================
colors = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3]
extra_colors = plt.cm.Set3(np.linspace(0, 1, 11))[:, :3]
all_colors = np.vstack(([[0,0,0]], colors, extra_colors))[:N_CLASSES]
custom_cmap = mcolors.ListedColormap(all_colors)

def mask_to_color(mask):
    """ Converts a segmentation mask (indices) to a colored RGB image. """
    color_mask = custom_cmap(mask / (N_CLASSES - 1))[:, :, :3]
    return (color_mask * 255).astype(np.uint8)

def create_legend(height, unique_classes):
    """
    Creates a white sidebar listing the fruits found in the image.
    """
    width = 250  # Width of the legend panel
    legend = np.ones((height, width, 3), dtype=np.uint8) * 255 # White background
    
    y_offset = 40
    cv2.putText(legend, "Detected Fruits:", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    
    if len(unique_classes) <= 1: # Only background found
        cv2.putText(legend, "None", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        return legend

    for cls_idx in unique_classes:
        if cls_idx == 0: continue # Skip background
        
        # Get color (RGB -> BGR for OpenCV)
        # We use the exact same logic as mask_to_color to match colors perfectly
        rgba = custom_cmap(cls_idx / (N_CLASSES - 1))
        color_rgb = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0]) 
        
        # Draw Color Box
        cv2.rectangle(legend, (15, y_offset-15), (35, y_offset+5), color_bgr, -1)
        cv2.rectangle(legend, (15, y_offset-15), (35, y_offset+5), (0,0,0), 1) # Black border
        
        # Draw Text
        fruit_name = class_to_fruit.get(cls_idx, f"Class {cls_idx}")
        cv2.putText(legend, fruit_name, (45, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        
        y_offset += 35
        
    return legend

# ==========================================
# 4. LOAD MODEL & PREPROCESS
# ==========================================
def load_model(path):
    print(f"Loading model from {path}...")
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=N_CLASSES
    )
    try:
        model.load_state_dict(torch.load(path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file '{path}' not found.")
        exit()
    model.to(device)
    model.eval()
    return model

test_transform = A.Compose([
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ==========================================
# 5. MAIN INFERENCE LOOP
# ==========================================
def run_inference():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    model = load_model(MODEL_PATH)
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {INPUT_FOLDER}")
        return

    print(f"Processing {len(image_files)} images...")

    for img_name in tqdm(image_files):
        img_path = os.path.join(INPUT_FOLDER, img_name)
        
        image = cv2.imread(img_path)
        if image is None: continue
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        augmented = test_transform(image=original_image)
        tensor_img = augmented["image"].unsqueeze(0).to(device)
        

        with torch.no_grad():
            output = model(tensor_img)
            pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()
            

        colored_mask = mask_to_color(pred_mask)
        save_mask = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)


        unique_classes = np.unique(pred_mask)
        legend_img = create_legend(IMG_HEIGHT, unique_classes)
        

        resized_original = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        

        final_output = np.hstack((resized_original, save_mask, legend_img))
        

        output_filename = os.path.splitext(img_name)[0] + "_mapped.png"
        save_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(save_path, final_output)

    print(f"Done! Check the '{OUTPUT_FOLDER}' folder for images with legends.")

if __name__ == "__main__":
    run_inference()
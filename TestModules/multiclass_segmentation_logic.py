import cv2
import numpy as np
import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class FruitMulticlassSegmenter:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_classes = 31
        self.img_height = 512
        self.img_width = 512
        
        # 1. Class Mappings
        self.fruit_to_class = {
            "background": 0, "apple_Gala":1, "apple_Golden Delicious":2, "Avocado":3, "Banana":4,
            "Berry":5, "Burmese Grape":6, "Carambola":7, "Date Palm":8, "Dragon":9, "Elephant Apple":10,
            "Grape":11, "Green Coconut":12, "Guava":13, "Hog Plum":14, "Kiwi":15, "Lichi":16, "Malta":17,
            "Mango Golden Queen":18, "Mango_Alphonso":19, "Mango_Amrapali":20, "Mango_Bari":21,
            "Mango_Himsagar":22, "Olive":23, "Orange":24, "Palm":25, "Persimmon":26, "Pineapple":27,
            "Pomegranate":28, "Watermelon":29, "White Pear":30
        }
        self.class_to_fruit = {v: k for k, v in self.fruit_to_class.items()}

        # 2. Color Map Setup
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3]
        extra_colors = plt.cm.Set3(np.linspace(0, 1, 11))[:, :3]
        self.all_colors = np.vstack(([[0,0,0]], colors, extra_colors))[:self.n_classes]
        self.custom_cmap = mcolors.ListedColormap(self.all_colors)

        # 3. Transforms
        self.transform = A.Compose([
            A.Resize(height=self.img_height, width=self.img_width),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        # 4. Load Model
        self.model = self._load_model(model_path)

    def _load_model(self, path):
        model = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            in_channels=3,
            classes=self.n_classes
        )
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def _mask_to_color(self, mask):
        color_mask = self.custom_cmap(mask / (self.n_classes - 1))[:, :, :3]
        return (color_mask * 255).astype(np.uint8)

    def _create_legend(self, unique_classes):
        width = 280  # Increased width slightly for better padding
        # Use a slight off-white background (Light Gray) instead of harsh white
        bg_color = (245, 245, 245) 
        legend = np.ones((self.img_height, width, 3), dtype=np.uint8)
        legend[:] = bg_color
    
        # Layout Config
        start_x = 20
        text_color = (40, 40, 40) # Dark gray text is softer than pure black
        font = cv2.FONT_HERSHEY_SIMPLEX
    
        # --- Header ---
        cv2.putText(legend, "Detected Fruits:", (start_x, 35), font, 0.8, text_color, 2, cv2.LINE_AA)
        # Add a separator line under the header
        cv2.line(legend, (start_x, 50), (width - start_x, 50), (200, 200, 200), 2)

        y_offset = 90
        box_size = 20  # Size of the color square
    
        # --- Handle Empty Case ---
        if len(unique_classes) <= 1: # Assuming class 0 is background
            cv2.putText(legend, "None detected", (start_x, y_offset), font, 0.7, (100, 100, 100), 1, cv2.LINE_AA)
            return legend

        # --- Draw Classes ---
        for cls_idx in unique_classes:
            if cls_idx == 0: continue # Skip background
        
            # Color processing
            rgba = self.custom_cmap(cls_idx / (self.n_classes - 1))
            color_bgr = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))
        
            # 1. Draw the Color Box (Filled)
            cv2.rectangle(legend, (start_x, y_offset - box_size), 
                          (start_x + box_size, y_offset), color_bgr, -1)
        
            # 2. Draw Box Border (for contrast against light colors)
            cv2.rectangle(legend, (start_x, y_offset - box_size), 
                      (start_x + box_size, y_offset), (180, 180, 180), 1)
        
            # 3. Draw Text (Vertically centered relative to the box)
            fruit_name = self.class_to_fruit.get(cls_idx, f"Class {cls_idx}")
        
            # Get text size to align it perfectly
            (text_w, text_h), baseline = cv2.getTextSize(fruit_name, font, 0.6, 1)
        
            # Calculate text Y position so it centers with the box
            text_y = y_offset - (box_size // 2) + (text_h // 2)
        
            cv2.putText(legend, fruit_name, (start_x + box_size + 15, text_y), 
                        font, 0.65, text_color, 1, cv2.LINE_AA)
        
            y_offset += 45 # Increased spacing between rows

        return legend

    def predict(self, image_path):
        """
        Processes an image and returns the final visualization (original + mask + legend).
        """
        image = cv2.imread(image_path)
        if image is None: return None
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        augmented = self.transform(image=rgb_image)
        tensor_img = augmented["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor_img)
            pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()

        # Generate visuals
        colored_mask = self._mask_to_color(pred_mask)
        save_mask = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
        legend_img = self._create_legend(np.unique(pred_mask))
        resized_original = cv2.resize(image, (self.img_width, self.img_height))

        # Horizontal Stack
        return np.hstack((resized_original, save_mask, legend_img))
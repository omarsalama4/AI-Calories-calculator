import os
import cv2
import torch
from PIL import Image
import re

from TestModules.binary_classification_logic import FoodFruitClassifier
from TestModules.fruit_classifier import OZnetClassifier
from TestModules.siamese_logic import SwinProtoNetSiamese
from TestModules.binary_segmentation_logic import FruitBinarySegmenter
from TestModules.multiclass_segmentation_logic import FruitMulticlassSegmenter

BASE_TEST_DIR = "Test Cases Structure"
INTEGRATED_DIR = os.path.join(BASE_TEST_DIR, "Integerated Test")
SIAMESE_DIR = os.path.join(BASE_TEST_DIR, "Siamese Case II Test")
OUTPUT_DIR = "test_results/Final_Integrated_Run"


FRUIT_CAL_FILE = "TestModules\Fruit_cal.txt"
FOOD_CAL_FILE = "TestModules\Food_cal.txt"

MODEL_PATHS = {
    "binary_clf": "FoodFruitClassification/model_1.pth",
    "oz_subcategory": "FruitClassification/FruitClassification_1.pth",
    "few_shot": "One-Few shot model/SWIN_ProtoNet/swin_protonet_hyperparameters.pth",
    "bin_seg": "Fruit binary Segmentation/best_fruit_segmentation.pth",
    "multi_seg": "Fruit Multiclass Segmentation/best_model_multiclass.pth"
}

def load_calorie_map(fruit_file, food_file):
    cal_map = {}
    for file_path in [fruit_file, food_file]:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    if ':' in line and 'calories per gram' in line:
                        parts = line.split(':')
                        name = parts[0].strip()
                        val_str = parts[1].replace('~', '').split('calories')[0].strip()
                        try:
                            cal_map[name] = float(val_str)
                        except ValueError:
                            continue
    return cal_map

def get_grams(text: str) -> int:
    match = re.search(r'(\d+)g', text)   
    if match:
        return int(match.group(1)) 
    return 0

def run_tests():

    calorie_database = load_calorie_map(FRUIT_CAL_FILE, FOOD_CAL_FILE)
    
    print("Initializing models...")
    binary_clf = FoodFruitClassifier(MODEL_PATHS["binary_clf"])
    subcategory_clf = OZnetClassifier(MODEL_PATHS["oz_subcategory"])
    siamese_engine = SwinProtoNetSiamese(MODEL_PATHS["few_shot"])
    bin_segmenter = FruitBinarySegmenter(MODEL_PATHS["bin_seg"])
    multi_segmenter = FruitMulticlassSegmenter(MODEL_PATHS["multi_seg"])


    print("\n--- Starting Integrated Test ---")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    img_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    integrated_images = [f for f in os.listdir(INTEGRATED_DIR) if f.lower().endswith(img_extensions)]

    for img_name in integrated_images:
        img_path = os.path.join(INTEGRATED_DIR, img_name)
        img_id = os.path.splitext(img_name)[0]

        img_output_dir = os.path.join(OUTPUT_DIR, img_id)
        os.makedirs(img_output_dir, exist_ok=True)

        main_class, _ = binary_clf.predict(img_path)
        if main_class == "Fruit":
            sub_class, _ = subcategory_clf.predict(img_path)
            cal_per_gram = calorie_database.get(sub_class, 0.0)
            grams = get_grams(img_path)
            calorie_line = f"{cal_per_gram * grams} calories"
        else:
            sub_class = "General Food"
            calorie_line = "Calorie data not calculated for general food"

        text_file_path = os.path.join(img_output_dir, f"{img_id}_results.txt")
        with open(text_file_path, "w") as f:
            f.write(f"{main_class}\n")
            f.write(f"{sub_class}\n")
            f.write(f"{calorie_line}\n")

        if main_class == "Fruit":
            bin_mask_viz = bin_segmenter.predict(img_path)
            if bin_mask_viz is not None:
                cv2.imwrite(os.path.join(img_output_dir, "binary_mask.png"), bin_mask_viz)
            
            multi_mask_viz = multi_segmenter.predict(img_path)
            if multi_mask_viz is not None:
                cv2.imwrite(os.path.join(img_output_dir, "multi_mask.png"), multi_mask_viz)
        
        print(f"Processed: {img_id}")


    print("\n--- Starting Siamese Case II Test ---")
    if os.path.exists(SIAMESE_DIR):
        most_similar_name, score = siamese_engine.predict_most_similar(SIAMESE_DIR)
        print(f"SIAMESE RESULT: Most similar is '{most_similar_name}' (Score: {score:.4f})")


if __name__ == "__main__":
    run_tests()
import streamlit as st
import os
import cv2
import torch
import numpy as np
from PIL import Image
import time
import re
from TestModules.binary_classification_logic import FoodFruitClassifier
from TestModules.fruit_classifier import OZnetClassifier
from TestModules.siamese_logic import SwinProtoNetSiamese
from TestModules.binary_segmentation_logic import FruitBinarySegmenter
from TestModules.multiclass_segmentation_logic import FruitMulticlassSegmenter
from TestModules.FewShotSubCat import SwinProtoNetClassifier

st.set_page_config(
    page_title="Yummy AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .report-box {
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models():
    """
    Loads all models once and caches them in memory.
    Handles prototype computation and caching for the few-shot model.
    """
    paths = {
        "binary_clf": "FoodFruitClassification/model_1.pth",
        "oz_subcategory": "FruitClassification/FruitClassification_1.pth",
        "few_shot": "One-Few shot model/SWIN_ProtoNet/swin_protonet_hyperparameters.pth",
        "bin_seg": "Fruit binary Segmentation/best_fruit_segmentation.pth",
        "multi_seg": "Fruit Multiclass Segmentation/best_model_multiclass.pth",
        "few_shot_subcat": "One-Few shot model/SWIN_ProtoNet/swin_protonet_hyperparameters.pth"
    }
    

    for name, p in paths.items():
        if not os.path.exists(p):
            st.error(f"Missing model file: {p}")
            return None


    binary_clf = FoodFruitClassifier(paths["binary_clf"])
    oznet_clf = OZnetClassifier(paths["oz_subcategory"])
    siamese_clf = SwinProtoNetSiamese(paths["few_shot"])
    bin_seg_clf = FruitBinarySegmenter(paths["bin_seg"])
    multi_seg_clf = FruitMulticlassSegmenter(paths["multi_seg"]) 
    few_shot_subcat_clf = SwinProtoNetClassifier(paths["few_shot_subcat"])
    
    PROTO_CACHE_FILE = "cached_prototypes.pt"
    SUPPORT_FOLDER = "Data/FewShotData" 

    if os.path.exists(PROTO_CACHE_FILE):
        success = few_shot_subcat_clf.load_prototypes(PROTO_CACHE_FILE)
        if not success:
            if os.path.exists(SUPPORT_FOLDER):
                few_shot_subcat_clf.precompute_prototypes(SUPPORT_FOLDER, shots=3)
                few_shot_subcat_clf.save_prototypes(PROTO_CACHE_FILE)
    else:
        if os.path.exists(SUPPORT_FOLDER):
            few_shot_subcat_clf.precompute_prototypes(SUPPORT_FOLDER, shots=3)
            few_shot_subcat_clf.save_prototypes(PROTO_CACHE_FILE)
        else:
            st.warning(f"Support folder '{SUPPORT_FOLDER}' not found. Few-shot classification may fail.")

    models = {
        "binary": binary_clf,
        "oznet": oznet_clf,
        "siamese": siamese_clf,
        "bin_seg": bin_seg_clf,
        "multi_seg": multi_seg_clf,
        "few_shot_subcat": few_shot_subcat_clf
    }
    return models

@st.cache_data
def load_calorie_data():
    """Parses text files for calorie info."""
    cal_map = {}
    files = ["TestModules/Fruit_cal.txt", "TestModules/Food_cal.txt"]
    
    for f_path in files:
        if os.path.exists(f_path):
            with open(f_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        parts = line.split(':')
                        name = parts[0].strip().lower()
                        val = re.split(r'\s*calories?', parts[1].replace('~', ''), flags=re.IGNORECASE)[0].strip()
                        try:
                            cal_map[name] = float(val)
                        except:
                            continue

    print(len(cal_map))
    return cal_map


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("temp", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return os.path.join("temp", uploaded_file.name)
    except Exception as e:
        return None

def get_grams(text: str) -> int:
    match = re.search(r'(\d+)g', text)   
    if match:
        return int(match.group(1)) 
    return 0

def replace_special_chars(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return str(cleaned_text)




def main(): 
    st.sidebar.title("🍎 Yummy AI")
    st.sidebar.markdown("---")
    app_mode = st.sidebar.radio("Select Mode", 
        ["Single Image Analysis", "Batch Integrated Test", "Similarity Matching"])

    st.sidebar.info("System Status: Models Loaded ✅")
    
    models = load_all_models()
    calorie_map = load_calorie_data()
    
    if not os.path.exists("temp"):
        os.makedirs("temp")

    if app_mode == "Single Image Analysis":
        st.title("🔬 Single Image Analysis")
        st.markdown("Upload an image to detect if it is Food or Fruit, calculate calories, and generate segmentation masks.")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 2])
            with col1:

                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                if st.button("Run Analysis"):
                    with st.spinner('Analyzing image...'):
                        temp_path = save_uploaded_file(uploaded_file)
                        
                        main_class, conf = models["binary"].predict(temp_path)
                        
                        st.markdown(f"### Detection Result: **{main_class}**")
                        st.progress(conf)
                        
                        if main_class == "Food":
                            sub_cls= models["few_shot_subcat"].predict(temp_path)
                            sub_cls = sub_cls.lower()
                            sub_cls = replace_special_chars(sub_cls)
                            cal_val = calorie_map.get(sub_cls, 0.0)
                            cal_str = f"{cal_val} cal/g" if cal_val != "Unknown" else "Data unavailable"
                            st.markdown(f"""
                            <div class='report-box'>
                                <b>1. Classification:</b> {main_class}<br>
                                <b>2. Subcategory:</b> {sub_cls}<br>
                                <b>3. Calories:</b> {cal_str}
                            </div>
                            """, unsafe_allow_html=True)
                            
                        elif main_class == "Fruit":
                            sub_class, sub_conf = models["oznet"].predict(temp_path)
                            sub_class = sub_class.lower()
                            sub_class = replace_special_chars(sub_class)
                            cal_val = calorie_map.get(sub_class, "Unknown")
                            cal_str = f"{cal_val} cal/g" if cal_val != "Unknown" else "Data unavailable"
                            
                            st.markdown(f"""
                            <div class='report-box'>
                                <b>1. Classification:</b> Fruit ({conf*100:.1f}%)<br>
                                <b>2. Subcategory:</b> {sub_class} ({sub_conf*100:.1f}%)<br>
                                <b>3. Calories:</b> {cal_str}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("### 🧬 Segmentation Analysis")
                            seg_col1, seg_col2 = st.columns(2)
                            
                            bin_viz = models["bin_seg"].predict(temp_path)
                            if bin_viz is not None:
                                with seg_col1:
                                    st.image(bin_viz, caption="Binary Segmentation (Region)", use_container_width=True)
                            
                            multi_viz = models["multi_seg"].predict(temp_path)
                            if multi_viz is not None:
                                with seg_col2:
                                    st.image(multi_viz, caption="Multiclass Segmentation (Type)", use_container_width=True)


    elif app_mode == "Batch Integrated Test":
        st.title("📂 Batch Integrated Test")
        st.markdown("Process an entire folder of images. Generates directories with reports and masks automatically.")
        
        folder_path = st.text_input("Enter Input Folder Path:", "Test Cases Structure/Integerated Test")
        output_path = st.text_input("Enter Output Folder Path:", "test_results/Streamlit_Run")
        
        if st.button("Start Batch Processing"):
            if not os.path.exists(folder_path):
                st.error("Input folder does not exist.")
            else:
                image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results_log = []
                
                for i, img_name in enumerate(image_files):
                    img_path = os.path.join(folder_path, img_name)
                    img_id = os.path.splitext(img_name)[0]
                    save_dir = os.path.join(output_path, img_id)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    main_cls, _ = models["binary"].predict(img_path)
                    sub_cls = "N/A"
                    cal_line = "N/A"
                    
                    if main_cls == "Fruit":
                        sub_cls, _ = models["oznet"].predict(img_path)
                        cal = calorie_map.get(sub_cls.lower(), 0.0)
                        grams = get_grams(img_path)
                        cal_line = f"{cal * grams} calories"
                        
                        bin_viz = models["bin_seg"].predict(img_path)
                        if bin_viz is not None:
                            cv2.imwrite(os.path.join(save_dir, "binary_mask.png"), bin_viz)
                            
                        multi_viz = models["multi_seg"].predict(img_path)
                        if multi_viz is not None:
                            cv2.imwrite(os.path.join(save_dir, "multi_mask.png"), multi_viz)
                    else:
                        sub_cls = models["few_shot_subcat"].predict(img_path)
                        sub_cls = replace_special_chars(sub_cls)
                        cal = calorie_map.get(sub_cls.lower(), 0.0)
                        grams = get_grams(img_path)
                        cal_line = f"{cal * grams} calories"

                    
                    with open(os.path.join(save_dir, "results.txt"), "w") as f:
                        f.write(f"{main_cls}\n{sub_cls}\n{cal_line}\n")
                    
                    results_log.append({
                        "Image": str(img_name), 
                        "Class": str(main_cls), 
                        "Sub": str(sub_cls),
                        "Calories": str(cal_line)
                    })

                    progress_bar.progress((i + 1) / len(image_files))
                    status_text.text(f"Processing {img_name}...")
                
                st.success(f"Batch processing complete! Results saved to {output_path}")
                st.dataframe(results_log)


    elif app_mode == "Similarity Matching":
        st.title("Similarity Matching Test")
        st.markdown("Identify the most similar reference image to an **Anchor** image in a folder.")
        
        siamese_folder = st.text_input("Enter Folder Path (must contain 'anchor' image):", "Test Cases Structure/Siamese Case II Test")
        
        if st.button("Find Match"):
            if not os.path.exists(siamese_folder):
                st.error("Folder not found.")
            else:
                with st.spinner("Comparing embeddings..."):
                    best_match, score = models["siamese"].predict_most_similar(siamese_folder)
                
                if best_match:
                    st.success("Match Found!")
                    
                    files = os.listdir(siamese_folder)
                    anchor_file = next((f for f in files if "anchor" in f.lower()), sorted(files)[0])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("⚓ Anchor Image")
                        st.image(os.path.join(siamese_folder, anchor_file), use_container_width=True)
                    
                    with col2:
                        st.subheader("✅ Best Match")
                        st.image(os.path.join(siamese_folder, best_match), use_container_width=True)
                        st.metric("Similarity Score", f"{score:.4f}")
                else:
                    st.warning("No valid images found in the directory.")

if __name__ == "__main__":
    main()
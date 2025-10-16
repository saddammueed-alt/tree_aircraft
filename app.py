# app.py
import streamlit as st
from ultralytics import YOLO
import os
import pandas as pd
from glob import glob
import shutil

# ==========================
# CONFIGURATION
# ==========================
TREE_MODEL_PATH = "models/tree_best.pt"
AIRCRAFT_MODEL_PATH = "models/aircraft_best.pt"

# Lazy load models
@st.cache_resource
def load_models():
    tree_model = YOLO(TREE_MODEL_PATH)
    aircraft_model = YOLO(AIRCRAFT_MODEL_PATH)
    return tree_model, aircraft_model

tree_model, aircraft_model = load_models()

# ==========================
# STREAMLIT UI
# ==========================
st.set_page_config(page_title="Trees & Aircraft Detection", layout="centered")

st.image("bg2.png", use_container_width=True)
st.title("🌲 Trees & ✈️ Aircraft Detection & Counting")

# Task Selection
task = st.selectbox("Select Task", ["Tree Detection", "Aircraft Detection"])

# Folder Inputs
input_folder = st.text_input("📁 Input Folder Path", "")
output_folder = st.text_input("💾 Output Folder Path", "")

confidence = st.slider("Detection Confidence", 0.0, 1.0, 0.1, 0.05)

run_button = st.button("🚀 Run Detection")

if run_button:
    if not input_folder or not output_folder:
        st.error("Please provide both input and output folder paths.")
    else:
        with st.spinner(f"Running {task}..."):
            # Select model
            if task == "Tree Detection":
                model = tree_model
                imgsz = (256, 256)
                save_crop = True
                show_labels = False
                project_name = "predict"
            else:
                model = aircraft_model
                imgsz = 1024
                save_crop = True
                show_labels = True
                project_name = "predictions"

            # Run YOLO inference
            model.predict(
                source=input_folder,
                save=True,
                project=output_folder,
                name=project_name,
                imgsz=imgsz,
                conf=confidence,
                line_width=2,
                show_conf=False,
                save_crop=save_crop,
                save_txt=True,
                show_labels=show_labels,
                exist_ok=True,
                classes=[0]
            )

            # Results
            predict_folder = os.path.join(output_folder, project_name)
            label_dir = os.path.join(predict_folder, "labels")

            if not os.path.exists(label_dir):
                total_objects = 0
            else:
                label_files = glob(os.path.join(label_dir, "*.txt"))
                total_objects = sum(len(open(f).readlines()) for f in label_files)

            if task == "Tree Detection":
                cropped_folder = os.path.join(output_folder, "cropped")
                os.makedirs(cropped_folder, exist_ok=True)
                cropped_images = glob(os.path.join(predict_folder, "crops", "*", "*.jpg"))
                for img in cropped_images:
                    shutil.move(img, os.path.join(cropped_folder, os.path.basename(img)))

            # Save results to Excel
            if os.path.exists(label_dir):
                label_files = glob(os.path.join(label_dir, "*.txt"))
                df = pd.DataFrame({
                    "Image": [os.path.basename(f).replace('.txt', '') for f in label_files],
                    f"{task.split()[0]} Count": [len(open(f).readlines()) for f in label_files]
                })
                df.loc["Total"] = ["Grand Total", total_objects]
                out_path = os.path.join(output_folder, f"{task.lower().replace(' ', '_')}_counts.xlsx")
                df.to_excel(out_path, index=False)
                st.success(f"{task} complete! Total count: {total_objects}")
                st.download_button("📥 Download Excel Results", open(out_path, "rb"), file_name=os.path.basename(out_path))
            else:
                st.warning("No detections found.")

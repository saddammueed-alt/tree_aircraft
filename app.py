import streamlit as st
from ultralytics import YOLO
import os
import shutil
import pandas as pd
from glob import glob
import zipfile
import tempfile
from PIL import Image

# ==========================
# MODEL PATHS
# ==========================
TREE_MODEL_PATH = "models/tree_best.pt"
AIRCRAFT_MODEL_PATH = "models/aircraft_best.pt"

# ==========================
# LOAD MODELS (cached)
# ==========================
@st.cache_resource
def load_models():
    tree_model = YOLO(TREE_MODEL_PATH)
    aircraft_model = YOLO(AIRCRAFT_MODEL_PATH)
    return tree_model, aircraft_model

tree_model, aircraft_model = load_models()

# ==========================
# APP SETUP
# ==========================
st.set_page_config(page_title="Trees & Aircraft Detection", layout="centered")
st.title("🌲 Trees & ✈️ Aircraft Detection & Counting")

st.markdown("Upload your **images** or a **ZIP folder** of images below to run detection.")

# ==========================
# TASK SELECTION
# ==========================
task = st.selectbox("Select Task", ["Tree Detection", "Aircraft Detection"])
confidence = st.slider("Detection Confidence", 0.0, 1.0, 0.1, 0.05)

uploaded_files = st.file_uploader(
    "📁 Upload Images or a ZIP file",
    type=["jpg", "jpeg", "png", "zip"],
    accept_multiple_files=True
)

run_button = st.button("🚀 Run Detection")

if run_button:
    if not uploaded_files:
        st.error("Please upload at least one image or ZIP file.")
    else:
        with st.spinner(f"Running {task}... Please wait ⏳"):
            # Create temporary directories
            input_dir = tempfile.mkdtemp()
            output_dir = tempfile.mkdtemp()

            # Handle uploaded files
            for file in uploaded_files:
                if file.name.endswith(".zip"):
                    zip_path = os.path.join(input_dir, file.name)
                    with open(zip_path, "wb") as f:
                        f.write(file.getvalue())
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(input_dir)
                else:
                    img_path = os.path.join(input_dir, file.name)
                    with open(img_path, "wb") as f:
                        f.write(file.getvalue())

            # Choose model parameters
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
                source=input_dir,
                save=True,
                project=output_dir,
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

            # Count detections
            predict_folder = os.path.join(output_dir, project_name)
            label_dir = os.path.join(predict_folder, "labels")

            if not os.path.exists(label_dir):
                total_objects = 0
                st.warning("No detections found.")
            else:
                label_files = glob(os.path.join(label_dir, "*.txt"))
                total_objects = sum(len(open(f).readlines()) for f in label_files)

                # Excel results
                df = pd.DataFrame({
                    "Image": [os.path.basename(f).replace('.txt', '') for f in label_files],
                    f"{task.split()[0]} Count": [len(open(f).readlines()) for f in label_files]
                })
                df.loc["Total"] = ["Grand Total", total_objects]
                excel_path = os.path.join(output_dir, f"{task.lower().replace(' ', '_')}_counts.xlsx")
                df.to_excel(excel_path, index=False)

                st.success(f"{task} completed ✅ — Total count: {total_objects}")
                st.download_button("📊 Download Excel Results", open(excel_path, "rb"), file_name=os.path.basename(excel_path))

                # Display detections
                st.markdown("### 🔍 Detection Results")
                result_images = glob(os.path.join(predict_folder, "*.jpg"))
                for img_path in result_images[:10]:  # show first 10 images
                    st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)

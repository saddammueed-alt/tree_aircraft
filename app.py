# app.py
import os
import shutil
import tempfile
import glob
import pandas as pd
from pathlib import Path
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# ==========================
# MODEL PATHS (Update these if needed)
# ==========================
import os

# Relative paths — models are in /models folder in repo
TREE_MODEL_PATH = os.path.join("models", "tree_best.pt")
AIRCRAFT_MODEL_PATH = os.path.join("models", "aircraft_best.pt")

# Load models once (Streamlit caches this if you use @st.cache_resource)
@st.cache_resource
def load_models():
    tree_model = YOLO(TREE_MODEL_PATH)
    aircraft_model = YOLO(AIRCRAFT_MODEL_PATH)
    return tree_model, aircraft_model

tree_model, aircraft_model = load_models()

def run_detection(input_dir, output_dir, confidence, task):
    if task == "Tree Detection":
        model = tree_model
        imgsz = (256, 256)
        save_crop = True
        show_labels = False
        project_name = "predict"
    elif task == "Aircraft Detection":
        model = aircraft_model
        imgsz = 1024
        save_crop = True
        show_labels = True
        project_name = "predictions"
    else:
        raise ValueError("Invalid task")

    # Run YOLO prediction
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
    predict_folder = Path(output_dir) / project_name
    label_dir = predict_folder / "labels"
    total_count = 0

    if label_dir.exists():
        label_files = list(label_dir.glob("*.txt"))
        total_count = sum(len(open(f).readlines()) for f in label_files)

    # Move crops for tree detection
    if task == "Tree Detection":
        cropped_folder = Path(output_dir) / "cropped"
        cropped_folder.mkdir(exist_ok=True)
        crop_paths = list((predict_folder / "crops").rglob("*.jpg"))
        for img_path in crop_paths:
            shutil.move(str(img_path), cropped_folder / img_path.name)

    # Save Excel
    if label_dir.exists():
        label_files = list(label_dir.glob("*.txt"))
        df = pd.DataFrame({
            "Image": [f.stem for f in label_files],
            f"{task.split()[0]} Count": [len(open(f).readlines()) for f in label_files]
        })
        df.loc["Total"] = ["Grand Total", total_count]
        excel_path = Path(output_dir) / f"{task.lower().replace(' ', '_')}_counts.xlsx"
        df.to_excel(excel_path, index=False)

    return total_count, project_name

# ==========================
# STREAMLIT UI
# ==========================
st.set_page_config(page_title="Trees & Aircraft Detection", layout="centered")
st.title("🌳✈️ Trees & Aircraft Detection and Counting")

# Sidebar or main UI for inputs
task = st.selectbox("Select Task", ["Tree Detection", "Aircraft Detection"])

uploaded_files = st.file_uploader(
    "Upload Images (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.1, 0.05)

if st.button("Run Detection"):
    if not uploaded_files:
        st.error("Please upload at least one image.")
    else:
        with st.spinner("Running detection..."):
            # Create temporary input directory
            with tempfile.TemporaryDirectory() as temp_input, tempfile.TemporaryDirectory() as temp_output:
                input_dir = Path(temp_input)
                output_dir = Path(temp_output)

                # Save uploaded files
                for file in uploaded_files:
                    with open(input_dir / file.name, "wb") as f:
                        f.write(file.getbuffer())

                # Run detection
                try:
                    total_count, project_name = run_detection(
                        str(input_dir),
                        str(output_dir),
                        confidence,
                        task
                    )
                except Exception as e:
                    st.error(f"Error during detection: {e}")
                    st.stop()

                # Display count
                obj_name = task.split()[0]
                st.success(f"✅ {task} complete!")
                st.metric(f"{obj_name} Count", total_count)

                # Show sample results (first 4 images)
                predict_folder = output_dir / project_name
                result_images = list(predict_folder.glob("*.jpg")) + list(predict_folder.glob("*.png"))
                if result_images:
                    st.subheader("Detection Results (Sample)")
                    cols = st.columns(min(4, len(result_images)))
                    for i, img_path in enumerate(result_images[:4]):
                        img = Image.open(img_path)
                        cols[i].image(img, use_column_width=True, caption=img_path.name)

                # Provide download for results
                # Create ZIP of output folder
                import zipfile
                zip_path = output_dir / "results.zip"
                with zipfile.ZipFile(zip_path, 'w') as zf:
                    for root, dirs, files in os.walk(output_dir):
                        for file in files:
                            if file == "results.zip":
                                continue
                            zf.write(os.path.join(root, file),
                                     os.path.relpath(os.path.join(root, file), output_dir))

                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="📥 Download All Results (ZIP)",
                        data=f,
                        file_name="detection_results.zip",
                        mime="application/zip"

                    )

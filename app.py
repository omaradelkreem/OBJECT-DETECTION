import streamlit as st
from PIL import Image
import torch
import tempfile
import os
import sys
import numpy as np
import pathlib

# Fix PosixPath bug on Windows
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

# Set the correct path to your YOLOv5 repository
YOLO_PATH = r"C:\Users\yousi\yolov5"  # Make sure this is the correct path
if not os.path.exists(YOLO_PATH):
    st.error(f"YOLOv5 repository not found at: {YOLO_PATH}")
    st.stop()

# Add YOLOv5 repo to Python path
sys.path.insert(0, str(YOLO_PATH))

try:
    from utils.general import non_max_suppression, check_img_size
    from models.common import DetectMultiBackend
    from utils.plots import Annotator, colors
    from utils.augmentations import letterbox
except ImportError as e:
    st.error(f"Failed to import YOLOv5 modules: {str(e)}")
    st.error("Make sure you've cloned the YOLOv5 repository and installed its requirements")
    st.stop()

import cv2

# Load YOLOv5 model
@st.cache_resource
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(YOLO_PATH, 'best.pt')
    if not os.path.exists(model_path):
        st.error(f"Model weights not found at: {model_path}")
        st.stop()
    model = DetectMultiBackend(model_path, device=device)
    stride = int(model.stride)
    imgsz = check_img_size(640, s=stride)
    return model, stride, imgsz, device

model, stride, imgsz, device = load_model()

st.title("Lung Nodule Detection with YOLOv5")
st.write("Upload a chest X-ray image to detect lung nodules")

uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload a chest X-ray image in JPG, JPEG, or PNG format"
)

if uploaded_file is not None:
    try:
        # Load and display original image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            image.save(tmp.name, format='JPEG')
            im0 = cv2.imread(tmp.name)
            
            if im0 is None:
                st.error("Failed to read the uploaded image")
                st.stop()
            
            # Preprocess image
            im = letterbox(im0, imgsz, stride=stride)[0]
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)
            im_tensor = torch.from_numpy(im).to(device).float()
            im_tensor /= 255.0
            if im_tensor.ndimension() == 3:
                im_tensor = im_tensor.unsqueeze(0)

            # Perform detection
            with st.spinner('Analyzing image for lung nodules...'):
                pred = model(im_tensor, augment=False, visualize=False)
                pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

            # Draw results
            annotator = Annotator(im0, line_width=2)
            detection_count = 0
            
            for det in pred:
                if det is not None and len(det):
                    det[:, :4] = det[:, :4].round()
                    for *xyxy, conf, cls in reversed(det):
                        detection_count += 1
                        label = f'Nodule {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))

            result_img = annotator.result()
            
            if detection_count > 0:
                st.success(f"Detected {detection_count} lung nodule(s)")
            else:
                st.info("No lung nodules detected")
                
            st.image(result_img, caption="Detection Result", use_column_width=True)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        # Clean up temporary file
        if 'tmp' in locals() and os.path.exists(tmp.name):
            os.unlink(tmp.name)
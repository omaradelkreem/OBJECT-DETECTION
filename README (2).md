# Lung Tumor Detection Using Machine Learning

This project applies deep learning to assist in the early diagnosis of lung cancer by automatically detecting pulmonary nodules in CT scans using a YOLOv5-based object detection model.

## 🚀 Project Overview

Lung cancer is a leading cause of cancer-related deaths, largely due to late diagnosis. Manual detection of lung nodules in CT scans is time-consuming and error-prone. This project aims to automate the detection process using a YOLOv5 object detection model trained on annotated CT images.

## 👨‍💻 Team Members

- **Mostafa Nasser** - 120210001  
- **Omar Abd-Elkareem** - 120210388  
- **Yousef Mohamed Amer** - 120210391  

## 📂 Dataset

- **Images**: 2D CT scan slices in `.bmp` format  
- **Annotations**: Bounding boxes in Pascal VOC `.xml` format  
- **Labels**: Converted to YOLO format (class, x_center, y_center, width, height)

## ⚙️ Methodology

### 1. Data Preparation
- Converted XML annotations to YOLO format.
- Organized images and labels into `train` and `val` directories.
- Defined dataset structure in a `data.yaml` file.

### 2. Model Training
- Environment: Google Colab, PyTorch
- Model: YOLOv5s (`yolov5s.pt`)
- Training Settings:
  - Image size: 1024×1024
  - Batch size: 8
  - Epochs: 50
  - Adjusted confidence threshold for small nodules

### 3. Evaluation
- **Precision**: 0.108  
- **Recall**: 0.375  
- **mAP@0.5**: 0.0664  
- **mAP@0.5:0.95**: 0.0336  

## 📈 Results

- **Strengths**: Effective at detecting small nodules.
- **Limitations**: Misses some large nodules; low confidence on some detections.
- **Qualitative Output**: Bounding boxes drawn and results saved in annotated images and `.txt` files.

## 🔍 Discussion

- **Annotation Issues**: Inaccurate/missing labels affected model performance.
- **Single-Class Model**: Detects only "nodule"; lacks malignancy classification.
- **Small Dataset**: Limited nodule diversity reduced generalization.
- **Model Simplicity**: YOLOv5s was used for speed at the expense of accuracy.

## 🔮 Future Work

- Improve annotation accuracy.
- Use advanced data augmentation.
- Upgrade to more powerful models (YOLOv5m/v5x, YOLOv8, DETR).
- Incorporate multi-class classification (benign vs malignant).

## 📁 Project Structure

```
lung-nodule-detection/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── data.yaml
├── train.py
├── detect.py
└── README.md
```

## 🛠️ Dependencies

- Python
- PyTorch
- YOLOv5
- OpenCV
- Google Colab (for training)

## 📸 Sample Output

Annotated images showing detected lung nodules with bounding boxes and confidence scores.

## 📝 License

This project is for academic and educational purposes only.

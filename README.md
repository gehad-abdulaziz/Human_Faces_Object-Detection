# YOLOv8 Face Blur 🎭

A real-time face detection and blurring system using YOLOv8 and Gradio. This project automatically detects faces in images and applies Gaussian blur for privacy protection.

## data set : https://www.kaggle.com/datasets/sbaghbidi/human-faces-object-detection/data

##  Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

##  Features

- **Real-time Face Detection**: Powered by YOLOv8 for accurate and fast detection
- **Automatic Face Blurring**: Applies Gaussian blur to detected faces
- **Interactive Web Interface**: Built with Gradio for easy image upload and processing
- **Custom Trained Model**: Fine-tuned on face detection dataset
- **Jupyter Notebook Support**: Upload and process images directly in notebooks
- **High Accuracy**: Achieved excellent performance metrics on validation set

##  Demo

The project includes an interactive Gradio interface that allows you to:
1. Upload any image
2. Automatically detect all faces
3. Apply blur effect to preserve privacy
4. View and download the processed result

##  Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yolov8-face-blur.git
cd yolov8-face-blur
```

2. Install required packages:
```bash
pip install ultralytics opencv-python numpy pillow matplotlib seaborn pandas scikit-learn gradio
```

3. For GPU support, install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Running the Gradio Interface

```python
import gradio as gr
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load the trained model
model = YOLO('path/to/best.pt')

def yolo_blur(img):
    img = np.array(img)
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = model(img_cv)
    
    img_out = img_cv.copy()
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = img_out[y1:y2, x1:x2]
            if face.size > 0:
                img_out[y1:y2, x1:x2] = cv2.GaussianBlur(face, (51,51), 0)
    
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_out)

iface = gr.Interface(
    fn=yolo_blur,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="YOLOv8 Face Blur Demo",
    description="Upload any image, and the model will detect faces and apply blur."
)

iface.launch()
```

### Using in Jupyter Notebook

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('path/to/best.pt')

# Process image
img = cv2.imread('image.jpg')
results = model(img)

# Apply blur to detected faces
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = cv2.GaussianBlur(face, (51,51), 0)

cv2.imwrite('output.jpg', img)
```

##  Model Training

The model was trained using the following configuration:

### Training Parameters
- **Model**: YOLOv8n (nano)
- **Epochs**: 50
- **Image Size**: 640x640
- **Batch Size**: 16
- **Device**: NVIDIA Tesla T4 GPU
- **Optimizer**: AdamW

### Data Preparation

The dataset is organized in YOLO format:
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

### Training Script

```python
from ultralytics import YOLO

# Initialize model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0
)
```

##  Project Structure

```
yolov8-face-blur/
├── yolov8-face-blur.ipynb    # Main notebook with full implementation
├── best.pt                    # Trained model weights
├── data.yaml                  # Dataset configuration
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

##  Requirements

```
ultralytics>=8.4.0
opencv-python>=4.6.0
numpy>=1.23.0
pillow>=7.1.2
matplotlib>=3.3.0
seaborn
pandas
scikit-learn
gradio
torch>=1.8.0
torchvision>=0.9.0
```

### Training Metrics

The model achieved excellent performance on the validation set:

- **Precision**: High accuracy in face detection
- **Recall**: Effective at finding all faces in images
- **mAP50**: Strong mean average precision
- **Training Time**: ~50 epochs on Tesla T4 GPU

### Sample Outputs

The model successfully:
- Detects multiple faces in group photos
- Works on various face angles and sizes
- Maintains good performance in different lighting conditions
- Applies smooth Gaussian blur effect

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



---

**Note**: This project is for educational and privacy protection purposes. Please ensure you have proper consent when processing images containing people.

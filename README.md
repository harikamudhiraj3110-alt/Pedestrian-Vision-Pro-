# 🚶 Pedestrian Vision Pro
### Embedded Night-Vision System for Pedestrian Detection

Pedestrian Vision Pro is a computer-vision-based night-vision pedestrian detection system designed to detect pedestrians in low-light and night-time images.

The project compares two pedestrian/object-detection approaches:

- **YOLOv2** – a CNN-based object detection model
- **Image enhancement + HOG-based pedestrian detection** – an OpenCV-based pipeline using gamma correction, histogram equalization, CLAHE, thresholding, and HOG

The main objective is to improve pedestrian detection when pedestrians are difficult to distinguish because of darkness or poor image quality.

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [System Workflow](#-system-workflow)
- [Detection Methods](#-detection-methods)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Using the Application](#-using-the-application)
- [Experimental Results](#-experimental-results)
- [Advantages](#-advantages)
- [Limitations](#-limitations)
- [Applications](#-applications)
- [Future Enhancements](#-future-enhancements)
- [Conclusion](#-conclusion)
- [Author](#-author)

---

## 🔍 About the Project

Pedestrian detection is an important component of intelligent transportation and driver-assistance systems.

During daytime conditions, computer vision models can generally detect pedestrians effectively. However, detection becomes more difficult at night because of:

- Low illumination
- Dark backgrounds
- Poor image quality
- Reduced visibility of pedestrian features
- Low contrast between pedestrians and their surroundings

This project investigates pedestrian detection in night-vision images using two different approaches.

### Approach 1 — YOLOv2

The project uses a pre-trained **YOLOv2** model through OpenCV's DNN module for object detection.

### Approach 2 — Image Enhancement + HOG

The second pipeline enhances the night image using:

1. Gamma correction
2. Grayscale conversion
3. Histogram equalization
4. CLAHE enhancement
5. Thresholding
6. HOG-based pedestrian detection

The detected pedestrian regions are then displayed with bounding boxes.

---

## ❗ Problem Statement

Conventional object detection models may have difficulty detecting pedestrians in very dark night-vision images.

The objective of this project is to investigate an image-processing-based detection pipeline that can improve pedestrian visibility and detection in low-light conditions.

---

## 🎯 Objectives

The major objectives are:

- Detect pedestrians from night-vision images.
- Improve visibility of pedestrians in low-light conditions.
- Implement YOLOv2-based object detection.
- Implement an alternative image-enhancement and HOG-based detection pipeline.
- Compare the detection behavior of the two approaches.
- Provide a simple graphical interface for testing images.
- Display detected pedestrians using bounding boxes.

---

## ✨ Features

- 🖼️ Upload night-vision images
- 🌙 Designed for low-light/night images
- 🤖 YOLOv2 object detection
- 👤 HOG-based pedestrian detection
- 🔆 Gamma correction for image enhancement
- 📊 Histogram equalization
- 🎨 CLAHE contrast enhancement
- 📦 Bounding-box visualization
- 🖥️ Simple Tkinter GUI
- 🧪 Multiple test images included
- ⚡ Easy execution using `run.bat`

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Main programming language |
| **OpenCV** | Image processing and computer vision |
| **OpenCV DNN** | YOLOv2 model execution |
| **YOLOv2** | CNN-based object detection |
| **HOG** | Pedestrian feature detection |
| **NumPy** | Numerical and image-array operations |
| **Imutils** | Image resizing and utility functions |
| **Tkinter** | Graphical user interface |
| **Pillow** | Image handling in testing code |

---

# 🔄 System Workflow

```text
                  ┌─────────────────────┐
                  │  Night Vision Image │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │     Upload Image    │
                  └──────────┬──────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       ▼
        ┌─────────────────┐     ┌────────────────────┐
        │     YOLOv2      │     │ Image Enhancement  │
        │ Object Detection│     │                    │
        └────────┬────────┘     │ Gamma Correction   │
                 │              │ Histogram Equal.   │
                 │              │ CLAHE              │
                 │              │ Thresholding       │
                 │              └─────────┬──────────┘
                 │                        │
                 │                        ▼
                 │              ┌────────────────────┐
                 │              │ HOG Pedestrian     │
                 │              │ Detection           │
                 │              └─────────┬──────────┘
                 │                        │
                 └───────────┬────────────┘
                             ▼
                  ┌─────────────────────┐
                  │ Detection Result    │
                  │ + Bounding Boxes    │
                  └─────────────────────┘
```

---

# 🤖 Detection Methods

## 1. YOLOv2 Detection

The project loads the YOLOv2 configuration and weights using OpenCV:

```python
cv2.dnn.readNetFromDarknet(
    'yolov2model/yolov2.cfg',
    'yolov2model/yolov2.weights'
)
```

The input image is converted into a blob using:

```python
cv.dnn.blobFromImage(
    image,
    1/255.0,
    (416, 416),
    swapRB=True,
    crop=False
)
```

The YOLOv2 network then produces detection results.

### Detection Process

```text
Input Image
     ↓
Resize / Blob Creation
     ↓
YOLOv2 CNN
     ↓
Object Predictions
     ↓
Confidence Filtering
     ↓
Non-Maximum Suppression
     ↓
Bounding Boxes
     ↓
Detected Image
```

The included YOLOv2 label file contains **80 object classes**, including person, bicycle, car, motorbike, bus, truck, dog, cat, chair, laptop, cell phone, and other COCO classes.

---

# 🔆 2. Image Enhancement + HOG Pedestrian Detection

The second approach first improves the visibility of the night image.

### Step 1 — Gamma Correction

Gamma correction is used to adjust image brightness.

The project uses:

```python
gamma = 3.5
```

The transformation is implemented using a lookup table.

```text
Dark Image
    ↓
Gamma Correction
    ↓
Enhanced Image
```

### Step 2 — Grayscale Conversion

```python
gray_img = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)
```

### Step 3 — Histogram Equalization

```python
gray_img_eqhist = cv2.equalizeHist(gray_img)
```

### Step 4 — CLAHE

```python
clahe = cv2.createCLAHE(clipLimit=20)
gray_img_clahe = clahe.apply(gray_img_eqhist)
```

### Step 5 — Thresholding

```python
th = 80
max_val = 255

ret, o3 = cv2.threshold(
    gray_img_clahe,
    th,
    max_val,
    cv2.THRESH_TOZERO
)
```

### Step 6 — HOG Pedestrian Detection

OpenCV's default people detector is used:

```python
hog = cv2.HOGDescriptor()

hog.setSVMDetector(
    cv2.HOGDescriptor_getDefaultPeopleDetector()
)
```

The detector searches for pedestrian regions and marks detected regions using bounding boxes.

---

# 🧩 Project Structure

```text
Pedestrian-Vision-Pro/
│
├── Main.py
├── test.py
├── yoloDetection.py
├── run.bat
├── JAVA.code-workspace
│
├── testImages/
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   ├── 4.png
│   ├── 5.png
│   └── test.png
│
└── yolov2model/
    ├── yolov2.cfg
    ├── yolov2.weights
    ├── yolov2-labels
    └── temp.png
```

### File Description

#### `Main.py`

Main application file. It creates the Tkinter GUI, loads YOLOv2, allows image upload, runs both detection pipelines, and displays results.

#### `yoloDetection.py`

Contains YOLO detection functions such as:

```text
detectObject()
labelsBoundingBoxes()
listBoundingBoxes()
displayImage()
```

#### `test.py`

Contains testing code for image enhancement and HOG pedestrian detection.

#### `run.bat`

Windows batch file used to start the application:

```bat
python Main.py
pause
```

#### `yolov2.cfg`

YOLOv2 network configuration file.

#### `yolov2.weights`

Pre-trained YOLOv2 model weights.

#### `yolov2-labels`

Contains the 80 YOLOv2 object-class labels.

#### `testImages/`

Contains the night-vision images used to test the system.

---

# 💻 Installation

## 1. Install Python

Install Python 3.x.

Check the installation:

```bash
python --version
```

or:

```bash
py --version
```

## 2. Clone the Repository

```bash
git clone https://github.com/harikamudhiraj3110-alt/Pedestrian-Vision-Pro-.git
cd Pedestrian-Vision-Pro-
```

## 3. Install Required Libraries

```bash
pip install opencv-python numpy imutils pillow
```

Tkinter is normally included with standard Python installations on Windows.

---

# ▶️ How to Run

### Method 1 — Using `run.bat`

On Windows, double-click:

```text
run.bat
```

### Method 2 — Using Terminal

```bash
python Main.py
```

---

# 🖥️ Using the Application

1. Start the application.
2. Select **Upload Night Vision Image**.
3. Choose an image from `testImages` or another location.
4. Select the YOLOv2 detection option to run YOLOv2.
5. Select the HAAR/AdaBoost-labelled detection option to run the enhancement + HOG pipeline.
6. View the resulting bounding boxes.
7. Select **Exit** to close the application.

> Note: The GUI/documentation uses the name “HAAR + AdaBoost” for the second option, while the implementation uses OpenCV's HOG people detector after image enhancement.

---

# 📊 Experimental Results

The project documentation describes testing with **six night-vision images**.

| Method | Reported observation |
|---|---|
| YOLOv2 | Pedestrians detected in 4 of 6 test images |
| Enhanced HOG pipeline | Pedestrians detected in all 6 test images before considering false detections |
| Enhanced HOG pipeline | Project documentation reports 80% accuracy after accounting for false detections |

These are project-specific experimental observations, not a general benchmark. Results can vary depending on the dataset, lighting conditions, camera quality, thresholds, and evaluation methodology.

---

# ✅ Advantages

## YOLOv2

- CNN-based object detection.
- Detects multiple object categories.
- Provides class labels and confidence values.
- Uses non-maximum suppression to reduce overlapping detections.

## Enhanced HOG Pipeline

- Designed around improving low-light image visibility.
- Uses gamma correction and contrast enhancement.
- Uses OpenCV's built-in pedestrian detector.
- Can detect pedestrians that may be difficult for YOLOv2 under the tested conditions.

---

# ⚠️ Limitations

- Primarily tested with still images.
- Does not currently provide real-time video detection.
- False-positive detections can occur.
- YOLOv2 is not specifically trained for this project's night-vision dataset.
- Performance can vary with lighting conditions.
- No comprehensive precision/recall/mAP evaluation pipeline is included.
- GUI is a basic Tkinter interface.
- YOLOv2 weights can make the repository large.

---

# 🚀 Future Enhancements

- Real-time webcam/video pedestrian detection.
- Evaluation with modern YOLO versions.
- Training/fine-tuning on dedicated night-time pedestrian datasets.
- Precision, recall, F1-score and mAP evaluation.
- FPS and inference-time measurement.
- Automatic image-enhancement parameter selection.
- Embedded deployment.
- Pedestrian warning/alert system.
- Improved graphical user interface.

---

# 🌐 Applications

Potential applications include:

- Driver-assistance systems
- Night-time pedestrian detection
- Intelligent transportation systems
- Road-safety monitoring
- Autonomous/assisted driving research
- Night surveillance
- Computer-vision research
- Pedestrian safety systems

---

# 🔬 Technical Pipeline

```text
                 NIGHT-VISION IMAGE
                         │
                         ▼
                  Image Acquisition
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
          YOLOv2 Path          Enhancement Path
              │                     │
              │              Gamma Correction
              │                     │
              │              Grayscale Conversion
              │                     │
              │              Histogram Equalization
              │                     │
              │                    CLAHE
              │                     │
              │                 Thresholding
              │                     │
              │                HOG Detector
              │                     │
              ▼                     ▼
        Object Detection      Pedestrian Detection
              │                     │
              └──────────┬──────────┘
                         ▼
                  Bounding Boxes
                         │
                         ▼
                   Result Display
```

---

# 📚 Main Algorithms and Concepts

### YOLOv2

YOLO (You Only Look Once) is a deep-learning object detection approach that predicts object bounding boxes and classes from an input image.

### HOG

Histogram of Oriented Gradients describes local shape information using gradient orientations and is commonly used for pedestrian detection.

### Gamma Correction

Gamma correction modifies image intensity values to make dark regions more visible.

### Histogram Equalization

Histogram equalization redistributes image intensity values to improve image contrast.

### CLAHE

Contrast Limited Adaptive Histogram Equalization enhances local contrast while limiting excessive amplification.

### Non-Maximum Suppression

NMS removes highly overlapping detection boxes and retains stronger detections.

---

# 📁 Model Information

The project uses:

```text
yolov2.cfg
yolov2.weights
yolov2-labels
```

The YOLO configuration contains 80 object classes. The Python inference code creates a `416 × 416` input blob before passing the image to the network.

---

# 🔧 Troubleshooting

### `ModuleNotFoundError`

Install the required packages:

```bash
pip install opencv-python numpy imutils pillow
```

### YOLO model not found

Make sure these files exist:

```text
yolov2model/
├── yolov2.cfg
├── yolov2.weights
└── yolov2-labels
```

### Image not loading

Check that:

- The image path is valid.
- The image format is supported.
- The selected image exists.

### Python command not recognized

Try:

```bash
py Main.py
```

or install Python and add it to the system PATH.

---

# 📌 Project Highlights

| Item | Details |
|---|---|
| **Project** | Pedestrian Vision Pro |
| **Type** | Computer Vision / Machine Learning |
| **Domain** | Pedestrian Detection & Road Safety |
| **Language** | Python |
| **Detection** | YOLOv2 + HOG |
| **Image Processing** | OpenCV |
| **GUI** | Tkinter |
| **Input** | Night-vision images |
| **Output** | Object/pedestrian bounding boxes |

---

# 🎓 Academic Project

This project demonstrates the application of computer vision and image-processing techniques to pedestrian detection under low-light conditions.

It combines a deep-learning object detector with an image-enhancement and classical computer-vision pipeline to investigate detection performance on night-vision images.

---

# 👤 Author

**Harika Mudhiraj**

GitHub:  
https://github.com/harikamudhiraj3110-alt

---

# 📄 License

This project is intended for educational and research purposes.

If you reuse the code, model files, or datasets, verify and follow the respective licenses and attribution requirements of those components.

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub.

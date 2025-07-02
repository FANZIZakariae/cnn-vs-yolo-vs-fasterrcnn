# 🧠 Object Detection Using CNN, Faster R-CNN & YOLOv3

This project explores, implements, and compares three major object detection architectures — **Classic CNN**, **Faster R-CNN**, and **YOLOv3** — using real-world datasets and modern deep learning tools.

---

## 📌 Project Overview

The goal of this project is to analyze and compare different object detection models in terms of **accuracy**, **speed**, and **practical usability**. It focuses on detecting and localizing multiple objects in an image, especially in real-world scenes like road traffic or domestic environments.

---

## 🚀 Models Implemented

### 🟦 CNN (Convolutional Neural Network)

* Classic CNN approach used as a baseline for comparison
* Demonstrates basic image recognition capabilities
* Not suitable for multi-object detection in complex scenes

### 🟥 Faster R-CNN

* Two-stage detector with region proposal network (RPN)
* High accuracy and localization performance
* Slower inference time, suitable for high-precision tasks

### 🟨 YOLOv3 (You Only Look Once)

* One-stage detector
* Real-time performance with good accuracy
* Ideal for real-world applications like surveillance or robotics

---

## 🧪 Results & Observations

| Model            | Accuracy                | Speed    | Notes                              |
| ---------------- | ----------------------- | -------- | ---------------------------------- |
| **CNN**          | ❌ Low on complex scenes | ✅ Fast   | Struggles with multi-object scenes |
| **Faster R-CNN** | ✅ High                  | ❌ Slower | Great for small object detection   |
| **YOLOv3**       | ✅ Good (balanced)       | ✅ Fast   | Best real-time trade-off           |

YOLOv3 provided the most stable learning curve and lower loss over time, making it well-suited for real-time systems despite minor classification inaccuracies.

---

## 🧰 Tech Stack

* 🧠 **TensorFlow** + **TensorFlow Hub**
* 🖼️ **OpenCV** for visualization
* ⚙️ **Google Colab** (GPU acceleration)
* ⚡ **FastAPI** for possible API integration

---

## 📊 Datasets Used

* **COCO**: Common Objects in Context — 330k+ images across 80 categories
* **PASCAL VOC 2007**: 20 categories, over 9k annotated images
* **ImageNet**: For transfer learning and feature extraction

---

## 💡 Key Features

* Real-time detection with YOLOv3
* Region-based detection with Faster R-CNN
* Visual result annotations (bounding boxes, class labels, confidence scores)
* Evaluation metrics: Intersection over Union (IoU), loss curves
* Image pre-processing, augmentation, and inference filtering

---

## 📷 Sample Outputs

| Model        | Sample Detection                                  |
| ------------ | ------------------------------------------------- |
| CNN          | Detected "chair" incorrectly (false positive)     |
| Faster R-CNN | Correct bounding box, misclassified objects       |
| YOLOv3       | Fast & reliable detection with moderate precision |

---

## 📈 Performance Evaluation

*YOLOv3 shows faster convergence and lower training loss compared to others*

---

## 📚 Future Work

* Explore more recent architectures like YOLOv7 or DETR
* Fine-tune models on custom datasets (e.g., medical, industrial)
* Deploy as a real-time web API using FastAPI or Flask
* Integrate with mobile or embedded platforms (e.g., Raspberry Pi + Coral TPU)


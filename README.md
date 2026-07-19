# 🧠 Brain Tumor Classification using Convolutional Neural Networks (CNN)

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---

# 📌 Overview

Brain Tumor Classification is a Deep Learning project that automatically classifies Brain MRI scans into one of four categories using a Convolutional Neural Network (CNN).

The application provides an interactive Streamlit interface where users can upload an MRI image and instantly receive the predicted tumor class along with the model's confidence score.

This project demonstrates an end-to-end Deep Learning workflow including:

- Data Preprocessing
- CNN Model Development
- Model Training
- Model Evaluation
- Prediction
- Streamlit Web Application
- Cloud Deployment

---

# 🚀 Live Demo

## 🌐 Streamlit Application

https://brain-tumor-classification-bmnjnasr2wt6bqzzvcymig.streamlit.app/

---

# 🎯 Features

- 🧠 Brain MRI Classification
- 📤 Upload MRI Images
- ⚡ Instant Predictions
- 📊 Confidence Score
- 🎨 Modern Streamlit User Interface
- 📱 Responsive Design
- ☁️ Cloud Deployment
- 🔄 Automatic Model Download using Google Drive
- 💻 CPU Compatible

---

# 🧠 Tumor Classes

The CNN classifies MRI scans into the following four categories:

| Class |
|--------|
| Glioma |
| Meningioma |
| No Tumor |
| Pituitary |

---

# 🏗️ Project Structure

```text
Brain-Tumor-Classification/
│
├── app.py                 # Streamlit application
├── model.py               # CNN Architecture
├── train.py                # Model Training
├── test.py                 # Model Evaluation
├── predict.py               # Prediction Script
│
├── outputs/
│   ├── confusion_matrix.png
│   └── classification_report.txt
│
├── predicted_images/
│   └── predictions.png
│
├── assets/
│   ├── home.png
│   ├── example1.png
│   ├── example2.png
│   └── random_10_test_samples.png
│
├── requirements.txt
├── runtime.txt
├── .gitignore
├── README.md
│
└── saved_models/
    └── best_model.pth     # Downloaded automatically from Google Drive
```

---

# 🏛️ CNN Architecture

The model consists of multiple deep learning layers including:

- Convolution Layers
- Batch Normalization
- ReLU Activation
- Max Pooling
- Dropout
- Fully Connected Layers
- Softmax Output Layer

The model is trained to classify MRI scans into four tumor categories.

---

# ⚙️ Technologies Used

## Programming Language

- Python

## Deep Learning

- PyTorch
- Torchvision

## Web Framework

- Streamlit

## Data Processing

- NumPy
- Pandas

## Visualization

- Matplotlib

## Image Processing

- Pillow
- OpenCV

## Model Download

- Google Drive
- gdown

---

# 📊 Dataset

The project uses a Brain MRI image dataset containing four classes:

- Glioma
- Meningioma
- No Tumor
- Pituitary

### Image Size

All MRI scans are resized to

```
224 × 224
```

before being passed to the CNN model.

---

# 🧪 Model Pipeline

```text
Brain MRI Image
        │
        ▼
Image Upload
        │
        ▼
Resize (224×224)
        │
        ▼
Tensor Conversion
        │
        ▼
CNN Model
        │
        ▼
Softmax Layer
        │
        ▼
Predicted Tumor Class
        │
        ▼
Confidence Score
```

---

# 📈 Model Evaluation

The trained model was evaluated using a separate test dataset.

Evaluation includes:

- Classification Accuracy
- Prediction Confidence
- Confusion Matrix

Example prediction output:

```
Prediction:
Glioma

Confidence:
98.74%
```

### 🔍 Predictions on Random Test Samples

Below are predictions made by the trained model on 10 randomly sampled images from the **test dataset**, showing the actual label vs. the predicted label along with the confidence score for each.

![Random Test Sample Predictions](assets/random_10_test_samples.png)

*The model correctly predicts most samples with high confidence, including harder cases such as MRI images with different orientations (sagittal, axial, coronal) and even a non-MRI CT scan.*

---

# 📦 Installation

## Clone Repository

```bash
git clone https://github.com/sriharshith-2006/Brain-Tumor-Classification.git
```

Move into project folder

```bash
cd Brain-Tumor-Classification
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

The application will automatically download the trained model during the first run.

---

# 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

The trained model is **not included** in the repository because GitHub has a file size limit of 100 MB.

Instead, the application downloads the trained model automatically from Google Drive when the app starts.

---

# 📸 Screenshots

## 🏠 Home Page

The landing page introduces the app, explains what it does, and highlights the key features — drag & drop upload, instant results, the four supported tumor classes, and the confidence score.

![Home Page](assets/home.png)

---

## 📤 Upload & Predict — Example 1 (No Tumor)

A sagittal MRI scan uploaded and analyzed by the model, correctly predicted as **No Tumor** with **100.00% confidence**.

![Upload and Predict — No Tumor Example](assets/example1.png)

---

## 📤 Upload & Predict — Example 2 (Meningioma)

An axial MRI scan uploaded and analyzed by the model, correctly predicted as **Meningioma** with **96.84% confidence**.

![Upload and Predict — Meningioma Example](assets/example2.png)

---

# 💻 Requirements

```
torch==2.7.1
torchvision==0.22.1
streamlit==1.49.1
numpy==2.2.6
pandas==2.3.2
matplotlib==3.10.6
scikit-learn==1.7.2
pillow==11.3.0
opencv-python-headless==4.12.0.88
gdown==5.2.0
```

---

# 🚀 Future Improvements

- Grad-CAM Visualization
- Better CNN Architecture
- Faster Inference
- Batch Image Prediction
- REST API using FastAPI
- Docker Deployment
- Mobile Responsive Interface
- Model Quantization
- Multi-language Support

---

# ⚠️ Disclaimer

This application is intended for **educational and research purposes only**.

It should **not** be used as a substitute for professional medical diagnosis or clinical decision-making.

Always consult qualified healthcare professionals for medical advice.

---

# 👨‍💻 Author

## Sriharshith Janga

**B.Tech – Artificial Intelligence and Data Science**

**IIIT Sri City**

### GitHub

https://github.com/sriharshith-2006

---

# ⭐ Support

If you found this project useful,

please consider giving this repository a ⭐ on GitHub.

It helps others discover the project and supports future improvements.

---

# 📄 License

This project is licensed under the **MIT License**.

Feel free to use, modify, and distribute it for educational and research purposes.

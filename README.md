# Deepfake Detection & Facial Preprocessing Pipeline

An end-to-end deep learning pipeline for detecting deepfake images using advanced facial alignment, MTCNN-based detection, and robust image preprocessing.

---

## 🚀 Overview

This project builds and evaluates a deepfake classification model leveraging real and fake facial imagery. It incorporates a precise preprocessing workflow designed to normalize face orientation, eliminate background noise, and standardize inputs for optimal model performance.

---

## 📊 Model Performance

- **Test Accuracy:** **88%**

---

## 🛠️ Key Features & Pipeline Architecture

1. **Dataset Integration:** Automatically handles and extracts data from the Kaggle Deepfake and Real Images dataset (`manjilkarki/deepfake-and-real-images`).
2. **MTCNN Face Detection:** Employs Multi-task Cascaded Convolutional Networks (MTCNN) to accurately detect faces and extract facial landmarks (left and right eyes).
3. **Affine Face Alignment:** Calculates eye angles to dynamically rotate, align, and crop faces to ensure uniform orientation across all samples.
4. **Standardized Resizing & Normalization:**
   - Resizes aligned facial crops to standard $224 \times 224$ dimensions.
   - Normalizes pixel values using ImageNet mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`).
5. **GPU Acceleration:** Fully optimized for CUDA-enabled environments (tested with NVIDIA RTX 4090 and TensorFlow 2.21).

---

## 📂 Project Structure

    deepfake-project/
    │
    ├── dataset_selections/      # Saved training and testing CSV selections
    ├── .venv/                   # Python virtual environment
    ├── notebooks/               # Jupyter notebooks for training and preprocessing
    └── README.md                # Project documentation

---

## ⚙️ Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone <https://github.com/meetjain6091/deepfake-detector>
   cd deepfake-project
   ```

2. **Activate the virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

To run the preprocessing pipeline or test individual face alignments, launch Jupyter Notebook:

```bash
jupyter notebook
```

Execute the cells to initialize MTCNN, parse images, and feed standardized batches into your model pipeline.

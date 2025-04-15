# Landslide-Detection-using-ResNet18

## Project Overview
This project performs **image classification** to detect **landslide-prone areas** using satellite images and a **ResNet18 deep learning model** in **PyTorch**. The goal is to develop a model that can accurately distinguish between **landslide** and **non-landslide** regions.

Additionally, this project served as a **learning experience** for applying **transfer learning, computer vision, and model evaluation** techniques using PyTorch and TorchVision.

---

## Dataset
The dataset contains satellite images organized into two classes:
- **`landslide/`** → Images showing areas affected by landslides.
- **`non-landslide/`** → Images showing areas without landslides.

Each image is labeled based on its folder name and used for training, validation, and testing.

## About the Data

- The input consists of **multi-spectral images** with shape `(128, 128, 14)` — a 128x128 pixel grid with **14 spectral bands**.
- These bands likely include various wavelengths such as visible, NIR (Near-Infrared), and other remote sensing channels.
- The data is stored in an **HDF5 file** under the key `'img'`.

### NDVI Calculation

- We use the **Red** (band 4) and **NIR** (band 8) channels to compute the **Normalized Difference Vegetation Index (NDVI)**:
  
  \[
  NDVI = \frac{NIR - Red}{NIR + Red}
  \]

- The resulting `NDVI` map helps visualize vegetation health and is combined with two other bands into a final array of shape `(1, 128, 128, 3)` for further analysis.

---

## Steps in Analysis

### 1. Data Preprocessing
The dataset is prepared using PyTorch's `ImageFolder` and transformations from `torchvision.transforms`:
- Resize and crop images to a consistent shape
- Normalize image pixel values
- Convert images to tensors
- Split into **train**, **validation**, and **test** sets

---

### 2. Model Architecture
We use **ResNet18**, a pre-trained Convolutional Neural Network from `torchvision.models`, and fine-tune it for binary classification:
- Replace the final fully connected layer
- Use `CrossEntropyLoss` as the loss function
- Optimize using the **Adam** optimizer

---

### 3. Training & Validation
The model is trained using:
- Mini-batch gradient descent
- Validation at each epoch to track accuracy and avoid overfitting
- Plotting **training and validation loss curves**

Metrics Evaluated:
- **Accuracy** on training and validation sets
- **Confusion matrix** on the test set

---

### 4. Testing & Evaluation
After training:
- Predictions are made on the test dataset
- A **confusion matrix** is plotted
- Sample predictions are visualized to verify model performance

---

### 5. Visualizations
- **Loss Curves** → Training vs Validation Loss over epochs
- **Confusion Matrix** → Evaluates true vs predicted labels
- **Prediction Samples** → Images with predicted and true labels

---

## Final Insights
- The ResNet18 model achieved **high accuracy** on the test set, proving effective for landslide detection.
- The **confusion matrix** shows good classification balance.
- Visual inspection of predictions confirms that the model generalizes well.
- Further improvements could be achieved with **data augmentation** and **deeper models** like ResNet50 or ResNet101.

---

## Libraries Used
- **PyTorch** – Deep Learning Framework  
- **TorchVision** – Pretrained models and transforms  
- **Matplotlib** – Visualization  
- **NumPy** – Numerical operations  
- **Scikit-learn** – Evaluation metrics (e.g., confusion matrix)

---

## How to Run the Code?

### Prerequisites
1. Install Python (preferably 3.7+)
2. Install required libraries using:

   ```bash
   pip install torch torchvision numpy matplotlib scikit-learn
